from datetime import datetime
import functools
from time import time
from typing import Tuple, Any, Mapping, Sequence, Union, Iterable, Generator
import json
from dataclasses import dataclass

from absl import app, flags, logging
import jax
import jax.numpy as jnp
from jax.tree_util import PyTreeDef
import numpy as np
import optax
import flax.linen as nn
import flax.typing
from flax.training import orbax_utils
from flax.training.train_state import TrainState
import orbax.checkpoint

from graphneuralpdesolver.experiments import DIR_EXPERIMENTS
from graphneuralpdesolver.autoregressive import AutoregressivePredictor, OperatorNormalizer
from graphneuralpdesolver.dataset import read_datasets, shuffle_arrays, normalize, unnormalize, Dataset
from graphneuralpdesolver.models.graphneuralpdesolver import GraphNeuralPDESolver, AbstractOperator, ToyOperator
from graphneuralpdesolver.utils import disable_logging, Array
from graphneuralpdesolver.metrics import mse, rel_l2_error, rel_l1_error


SEED = 44
NUM_DEVICES = jax.local_device_count()

FLAGS = flags.FLAGS
flags.DEFINE_string(name='datadir', default=None, required=True,
  help='Path of the folder containing the datasets'
)
flags.DEFINE_string(name='params', default=None, required=False,
  help='Path of the previous experiment containing the initial parameters'
)
flags.DEFINE_integer(name='resolution', default=128, required=False,
  help='Resolution of the physical discretization'
)
flags.DEFINE_string(name='experiment', default=None, required=True,
  help='Name of the experiment: {"E1", "E2", "E3", "WE1", "WE2", "WE3"'
)
flags.DEFINE_integer(name='batch_size', default=4, required=False,
  help='Size of a batch of training samples'
)
flags.DEFINE_integer(name='epochs', default=20, required=False,
  help='Number of training epochs'
)
flags.DEFINE_float(name='lr', default=1e-04, required=False,
  help='Training learning rate'
)
flags.DEFINE_float(name='lr_decay', default=None, required=False,
  help='The minimum learning rate decay in the cosine scheduler'
)
flags.DEFINE_integer(name='latent_size', default=128, required=False,
  help='Size of latent node and edge features'
)
flags.DEFINE_integer(name='unroll_steps', default=1, required=False,
  help='Number of steps for getting a noisy input and applying the model autoregressively'
)
flags.DEFINE_integer(name='direct_steps', default=1, required=False,
  help='Maximum number of time steps between input/output pairs during training'
)
flags.DEFINE_bool(name='verbose', default=False, required=False,
  help='If passed, training reports for batches are printed'
)
flags.DEFINE_bool(name='debug', default=False, required=False,
  help='If passed, the code is launched only for debugging purposes.'
)

@dataclass
class EvalMetrics:
  error_autoreg_l1: Sequence[Tuple[int, Sequence[float]]] = None
  error_autoreg_l2: Sequence[Tuple[int, Sequence[float]]] = None
  error_direct_l1: float = None
  error_direct_l2: float = None

DIR = DIR_EXPERIMENTS / datetime.now().strftime('%Y%m%d-%H%M%S.%f')

def train(key: flax.typing.PRNGKey, model: nn.Module, dataset: Dataset, epochs: int,
  params: flax.typing.Collection = None) -> TrainState:
  """Trains a model and returns the state."""

  # Samples
  sample_traj, sample_spec = dataset.sample
  _use_specs = (sample_spec is not None)
  sample_traj = jax.device_put(jnp.array(sample_traj))
  sample_spec = jax.device_put(jnp.array(sample_spec)) if _use_specs else None

  # Set constants
  num_samples_trn = dataset.nums['train']
  num_times = sample_traj.shape[1]
  num_grid_points = sample_traj.shape[2:4]
  num_vars = sample_traj.shape[-1]
  unroll_offset = FLAGS.unroll_steps * FLAGS.direct_steps
  assert num_samples_trn % FLAGS.batch_size == 0
  num_batches = num_samples_trn // FLAGS.batch_size
  assert FLAGS.batch_size % NUM_DEVICES == 0
  batch_size_per_device = FLAGS.batch_size // NUM_DEVICES

  # Store the initial time
  time_int_pre = time()

  # Set the normalization statistics
  stats_trj_mean = jax.device_put(jnp.array(dataset.mean_trn))
  stats_trj_std = jax.device_put(jnp.array(dataset.std_trn))
  stats_res_mean = jax.device_put(jnp.array(dataset.mean_res_trn))
  stats_res_std = jax.device_put(jnp.array(dataset.std_res_trn))

  # Initialzize the model or use the loaded parameters
  if params:
    variables = {'params': params}
  else:
    subkey, key = jax.random.split(key)
    model_init_kwargs = dict(
      u_inp=jnp.ones(shape=(FLAGS.batch_size, 1, *num_grid_points, num_vars)),
      ndt=1.,
      specs=(
        jnp.ones_like(sample_spec).repeat(FLAGS.batch_size, axis=0)
        if sample_spec is not None else None
      ),
    )
    variables = jax.jit(model.init)(subkey, **model_init_kwargs)

  # Calculate the total number of parameters
  n_model_parameters = np.sum(
  jax.tree_util.tree_flatten(
    jax.tree_map(
      lambda x: np.prod(x.shape).item(),
      variables['params']
    ))[0]
  ).item()
  logging.info(f'Total number of trainable paramters: {n_model_parameters}')

  # Define the permissible lead times and number of batches
  lead_times = jnp.arange(unroll_offset, num_times - FLAGS.direct_steps)
  num_lead_times = num_times - unroll_offset - FLAGS.direct_steps

  # Set up the optimization components
  criterion_loss = mse
  lr = optax.cosine_decay_schedule(
    init_value=FLAGS.lr,
    decay_steps=(FLAGS.epochs * num_batches),
    alpha=FLAGS.lr_decay,
  ) if FLAGS.lr_decay else FLAGS.lr
  tx = optax.inject_hyperparams(optax.adamw)(learning_rate=lr, weight_decay=1e-8)
  state = TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)

  # Define the autoregressive predictor
  normalizer = OperatorNormalizer(
    operator=model,
    stats_trj=(stats_trj_mean, stats_trj_std),
    stats_res=(stats_res_mean, stats_res_std),  # TMP :: TODO: Support longer residuals
  )
  predictor = AutoregressivePredictor(operator=normalizer, num_steps_direct=FLAGS.direct_steps)

  def _compute_loss(
    params: flax.typing.Collection, specs: Array,
    u_lag: Array, ndt: int, u_tgt: Array, num_steps_autoreg: int) -> Array:
    """Computes the prediction of the model and returns its loss."""

    variables = {'params': params}
    # Apply autoregressive steps
    u_inp = predictor.jump(
      variables=variables,
      specs=specs,
      u_inp=u_lag,
      num_jumps=num_steps_autoreg,
    )
    # Get the output
    # NOTE: using checkpointed version to avoid memory exhaustion
    # TRY: Change it back to model.apply to get more performance, although it is only one step..
    # u_prd = predictor._apply_operator(variables, specs=specs, u_inp=u_inp, ndt=ndt)

    _loss_inputs = normalizer.get_loss_inputs(
      variables=variables,
      specs=specs,
      u_inp=u_inp,
      u_tgt=u_tgt,
      ndt=ndt,
    )

    return criterion_loss(*_loss_inputs)

  def _get_noisy_input(
    params: flax.typing.Collection, specs: Array,
    u_lag: Array, num_steps_autoreg: int) -> Array:
    """Apply the model to the lagged input to get a noisy input."""

    variables = {'params': params}
    u_inp_noisy = predictor.jump(
      variables=variables,
      specs=specs,
      u_inp=u_lag,
      num_jumps=num_steps_autoreg,
    )

    return u_inp_noisy

  def _get_loss_and_grads(
    params: flax.typing.Collection, specs: Array,
    u_lag: Array, u_tgt: Array, ndt: int) -> Tuple[Array, PyTreeDef]:
    """
    Computes the loss and the gradients of the loss w.r.t the parameters.
    """

    # Split the unrolling steps randomly to cut the gradients along the way
    # MODIFY: Change to JAX-generated random number (reproducability)
    noise_steps = np.random.choice(FLAGS.unroll_steps+1)
    grads_steps = FLAGS.unroll_steps - noise_steps

    # Get noisy input
    u_inp = _get_noisy_input(
      params, specs, u_lag, num_steps_autoreg=noise_steps)
    # Use noisy input and compute gradients
    loss, grads = jax.value_and_grad(_compute_loss)(
      params, specs, u_inp, ndt, u_tgt, num_steps_autoreg=grads_steps)

    return loss, grads

  def _get_loss_and_grads_direct_step(
    state: TrainState, key: flax.typing.PRNGKey,
    specs: Array, u_lag: Array, u_tgt: Array, ndt: int,
  ) -> Tuple[Array, PyTreeDef]:
    # NOTE: INPUT SHAPES [batch_size_per_device * num_lead_times, ...]

    # Shuffle the input/outputs along the batch axis
    if _use_specs:
      specs, u_lag, u_tgt = shuffle_arrays(key, [specs, u_lag, u_tgt])
    else:
      u_lag, u_tgt = shuffle_arrays(key, [u_lag, u_tgt])

    # Split into num_lead_times chunks and get loss and gradients
    # -> [num_lead_times, batch_size_per_device, ...]
    specs = jnp.stack(jnp.split(specs, num_lead_times)) if _use_specs else None
    u_lag = jnp.stack(jnp.split(u_lag, num_lead_times))
    u_tgt = jnp.stack(jnp.split(u_tgt, num_lead_times))

    # Add loss and gradients for each mini batch
    def _update_loss_and_grads_lead_time(i, carry):
      _loss_carried, _grads_carried = carry
      _loss_lead_time, _grads_lead_time = _get_loss_and_grads(
        params=state.params,
        specs=(specs[i] if _use_specs else None),
        u_lag=u_lag[i],
        u_tgt=u_tgt[i],
        ndt=ndt,
      )
      _loss_updated = _loss_carried + _loss_lead_time / num_lead_times
      _grads_updated = jax.tree_map(
        lambda g_old, g_new: (g_old + g_new / num_lead_times),
        _grads_carried,
        _grads_lead_time
      )
      return _loss_updated, _grads_updated

    # Loop over lead_times
    _init_loss = 0.
    _init_grads = jax.tree_map(lambda p: jnp.zeros_like(p), state.params)
    loss, grads = jax.lax.fori_loop(
      lower=0,
      upper=num_lead_times,
      body_fun=_update_loss_and_grads_lead_time,
      init_val=(_init_loss, _init_grads)
    )

    return loss, grads

  @functools.partial(jax.pmap,
    in_axes=(None, 0, 0, None),
    out_axes=(None, None, None),
    axis_name="device",
  )
  def _train_one_batch(
    state: TrainState, trajs: Array, specs: Array,
    key: flax.typing.PRNGKey) -> Tuple[TrainState, Array, Array]:
    """Loads a batch, normalizes it, updates the state based on it, and returns it."""

    # Get input output pairs for all lead times
    # -> [num_lead_times, batch_size_per_device, ...]
    u_lag_batch = jax.vmap(
        lambda lt: jax.lax.dynamic_slice_in_dim(
          operand=trajs,
          start_index=(lt-unroll_offset), slice_size=1, axis=1)
    )(lead_times)
    u_tgt_batch = jax.vmap(
        lambda lt: jax.lax.dynamic_slice_in_dim(
          operand=trajs,
          start_index=(lt+1), slice_size=FLAGS.direct_steps, axis=1)
    )(lead_times)
    specs_batch = (specs[None, :, :]
      .repeat(repeats=num_lead_times, axis=0)
    ) if _use_specs else None

    # Concatenate lead times along the batch axis
    # -> [batch_size_per_device * num_lead_times, ...]
    u_lag_batch = u_lag_batch.reshape(
        (batch_size_per_device * num_lead_times), 1, *num_grid_points, -1)
    u_tgt_batch = u_tgt_batch.reshape(
        (batch_size_per_device * num_lead_times), FLAGS.direct_steps, *num_grid_points, -1)
    specs_batch = specs_batch.reshape(
        (batch_size_per_device * num_lead_times), -1) if _use_specs else None

    # Compute loss and gradient by mapping on the time axis
    # Same u_lag and specs, loop over ndt
    key, subkey = jax.random.split(key)
    subkeys = jnp.stack(jax.random.split(subkey, num=FLAGS.direct_steps))
    ndt_batch = 1 + jnp.arange(FLAGS.direct_steps)  # -> [direct_steps,]
    u_tgt_batch = jnp.expand_dims(u_tgt_batch, axis=2).swapaxes(0, 1)  # -> [direct_steps, ...]

    # Shuffle direct_steps
    # NOTE: Redundent because we apply gradients on the whole batch
    # key, subkey = jax.random.split(key)
    # ndt_batch, u_tgt_batch = shuffle_arrays(subkey, [ndt_batch, u_tgt_batch])

    # Add loss and gradients for each direct_step
    def _update_loss_and_grads_direct_step(i, carry):
      _loss_carried, _grads_carried = carry
      _loss_direct_step, _grads_direct_step = _get_loss_and_grads_direct_step(
        state=state,
        key=subkeys[i],
        specs=(specs_batch if _use_specs else None),
        u_lag=u_lag_batch,
        u_tgt=u_tgt_batch[i],
        ndt=ndt_batch[i],
      )
      _loss_updated = _loss_carried + _loss_direct_step / FLAGS.direct_steps
      _grads_updated = jax.tree_map(
        lambda g_old, g_new: (g_old + g_new / FLAGS.direct_steps),
        _grads_carried,
        _grads_direct_step,
      )
      return _loss_updated, _grads_updated

    # Loop over the direct_steps
    _init_loss = 0.
    _init_grads = jax.tree_map(lambda p: jnp.zeros_like(p), state.params)
    loss, grads = jax.lax.fori_loop(
      lower=0,
      upper=FLAGS.direct_steps,
      body_fun=_update_loss_and_grads_direct_step,
      init_val=(_init_loss, _init_grads)
    )

    # Synchronize loss and gradients
    grads = jax.lax.pmean(grads, axis_name="device")
    loss = jax.lax.pmean(loss, axis_name="device")

    # Apply gradients
    state = state.apply_gradients(grads=grads)

    return state, loss, grads

  def train_one_epoch(
    state: TrainState, batches: Iterable[Tuple[Array, Array]],
    key: flax.typing.PRNGKey) -> Tuple[TrainState, Array, Array]:
    """Updates the state based on accumulated losses and gradients."""

    # Loop over the batches
    loss_epoch = 0.
    grad_epoch = 0.
    for idx, batch in enumerate(batches):
      begin_batch = time()

      # Unwrap the batch
      batch = jax.tree_map(jax.device_put, batch)  # Transfer to device memory
      trajs, specs = batch

      # Split the batch between devices
      # -> [NUM_DEVICES, batch_size_per_device, ...]
      trajs = jnp.concatenate(jnp.split(jnp.expand_dims(
        trajs, axis=0), NUM_DEVICES, axis=1), axis=0)
      specs = jnp.concatenate(jnp.split(jnp.expand_dims(
        specs, axis=0), NUM_DEVICES, axis=1), axis=0) if _use_specs else None

      # Get loss and updated state
      subkey, key = jax.random.split(key)
      state, loss, grads = _train_one_batch(state, trajs, specs, subkey)
      loss_epoch += loss * FLAGS.batch_size / num_samples_trn
      grad_epoch += np.mean(jax.tree_util.tree_flatten(jax.tree_map(jnp.mean, jax.tree_map(jnp.abs, grads)))[0]) / num_batches

      time_batch = time() - begin_batch

      if FLAGS.verbose:
        logging.info('\t'.join([
          f'\t',
          f'BTCH: {idx+1:04d}/{num_batches:04d}',
          f'TIME: {time_batch:06.1f}s',
          f'RMSE: {np.sqrt(loss).item() : .2e}',
        ]))

    return state, loss_epoch, grad_epoch

  @functools.partial(jax.pmap,
      in_axes=(None, 0, 0, None, None, None),
      static_broadcasted_argnums=(5,))
  def _predict_trajectory_autoregressively(
      state: TrainState, specs: Array, u_inp: Array,
      stats_inp: Tuple[Array, Array], stats_tgt: Tuple[Array, Array],
      num_steps: int,
    ) -> Array:
    """
    Normalizes the input and predicts the trajectories autoregressively.
    The input dataset must be raw (not normalized).
    """

    # Normalize the input
    u_inp = normalize(u_inp, mean=stats_inp[0], std=stats_inp[1])
    # Get normalized predictions
    variables = {'params': state.params}
    rollout, _ = predictor.unroll(
      variables=variables,
      specs=specs,
      u_inp=u_inp,
      num_steps=num_steps,
    )
    # Denormalize the predictions
    rollout = unnormalize(rollout, mean=stats_tgt[0], std=stats_tgt[1])

    return rollout

  @functools.partial(jax.pmap, in_axes=(None, 0, 0))
  # TMP :: TODO: Update normalization
  def _evaluate_direct_prediction(
    state: TrainState, trajs: Array, specs: Array) -> Tuple[Array, Array]:

    # Inputs are of shape [batch_size_per_device, ...]

    # Set lead times
    lead_times = jnp.arange(num_times - FLAGS.direct_steps)
    num_lead_times = num_times - FLAGS.direct_steps

    # Get input output pairs for all lead times
    # -> [num_lead_times, batch_size_per_device, ...]
    u_inp = jax.vmap(
        lambda lt: jax.lax.dynamic_slice_in_dim(
          operand=trajs,
          start_index=(lt), slice_size=1, axis=1)
    )(lead_times)
    u_tgt = jax.vmap(
        lambda lt: jax.lax.dynamic_slice_in_dim(
          operand=trajs,
          start_index=(lt+1), slice_size=FLAGS.direct_steps, axis=1)
    )(lead_times)
    specs = (jnp.array(specs[None, :, :])
      .repeat(repeats=num_lead_times, axis=0)
    ) if _use_specs else None
    mean_inp = jax.vmap(
        lambda lt: jax.lax.dynamic_slice_in_dim(
          operand=stats_trj_mean,
          start_index=(lt), slice_size=1, axis=1)
    )(lead_times)
    std_inp = jax.vmap(
        lambda lt: jax.lax.dynamic_slice_in_dim(
          operand=stats_trj_std,
          start_index=(lt), slice_size=1, axis=1)
    )(lead_times)
    mean_tgt = jax.vmap(  # TMP :: TODO: support direct steps > 1
        lambda lt: jax.lax.dynamic_slice_in_dim(
          operand=stats_res_mean[0],
          start_index=(lt), slice_size=1, axis=1)
    )(lead_times)
    std_tgt = jax.vmap(  # TMP :: TODO: support direct steps > 1
        lambda lt: jax.lax.dynamic_slice_in_dim(
          operand=stats_res_std[0],
          start_index=(lt), slice_size=1, axis=1)
    )(lead_times)

    def get_direct_errors(lt, carry):
      err_l1_mean, err_l2_mean = carry
      def get_direct_prediction(ndt, forcing):
        # TMP :: TODO: Use normalizer
        u_inp_nrm = normalize(u_inp[lt], mean=mean_inp[lt], std=std_inp[lt])
        r_prd_nrm = model.apply(
          variables={'params': state.params},
          u_inp=u_inp_nrm,
          specs=(specs[lt] if _use_specs else None),
          ndt=ndt,
        )
        r_prd = unnormalize(r_prd_nrm, mean=mean_tgt[lt], std=std_tgt[lt])  # TMP :: TODO: support direct steps > 1
        u_prd = u_inp[lt] + r_prd
        return (ndt+1), u_prd
      _, u_prd = jax.lax.scan(
        f=get_direct_prediction,
        init=1, xs=None, length=FLAGS.direct_steps,
      )
      u_prd = u_prd.squeeze(axis=2).swapaxes(0, 1)
      err_l1_mean += jnp.sqrt(jnp.sum(jnp.power(rel_l1_error(u_prd, u_tgt[lt]), 2), axis=1)) / num_lead_times
      err_l2_mean += jnp.sqrt(jnp.sum(jnp.power(rel_l2_error(u_prd, u_tgt[lt]), 2), axis=1)) / num_lead_times

      return err_l1_mean, err_l2_mean

    # Get mean errors per each sample in the batch
    err_l1_mean, err_l2_mean = jax.lax.fori_loop(
      body_fun=get_direct_errors,
      lower=0,
      upper=num_lead_times,
      init_val=(
        jnp.zeros(shape=(batch_size_per_device,)),
        jnp.zeros(shape=(batch_size_per_device,)),
      )
    )

    return err_l1_mean, err_l2_mean

  def evaluate(
    state: TrainState, batches: Iterable[Tuple[Array, Array]],
    parts: Union[Sequence[int], int] = 1) -> EvalMetrics:
    """Evaluates the model on a dataset based on multiple trajectory lengths."""

    # Initialize the containers
    if isinstance(parts, int):
      parts = [parts]
    error_ar_l1_per_var = {p: [] for p in parts}
    error_ar_l2_per_var = {p: [] for p in parts}
    error_ar_l1 = {p: [] for p in parts}
    error_ar_l2 = {p: [] for p in parts}
    error_dr_l1 = []
    error_dr_l2 = []

    for batch in batches:
      # Unwrap the batch
      batch = jax.tree_map(jax.device_put, batch)  # Transfer to device memory
      trajs, specs = batch

      # Split the batch between devices
      # -> [NUM_DEVICES, batch_size_per_device, ...]
      trajs = jnp.concatenate(jnp.split(jnp.expand_dims(
        trajs, axis=0), NUM_DEVICES, axis=1), axis=0)
      specs = jnp.concatenate(jnp.split(jnp.expand_dims(
        specs, axis=0), NUM_DEVICES, axis=1), axis=0) if _use_specs else None

      # Evaluate direct prediction
      _error_dr_l1_batch, _error_dr_l2_batch = _evaluate_direct_prediction(
        state, trajs, specs,
      )
      # Re-arrange the sub-batches gotten from each device
      _error_dr_l1_batch = _error_dr_l1_batch.reshape(FLAGS.batch_size, 1)
      _error_dr_l2_batch = _error_dr_l2_batch.reshape(FLAGS.batch_size, 1)
      # Append the errors to the list
      error_dr_l1.append(_error_dr_l1_batch)
      error_dr_l2.append(_error_dr_l2_batch)

      # # Evaluate autoregressive prediction  # TMP
      # for p in parts:
      #   # Get a dividable full trajectory
      #   traj_length = num_times - (num_times % p)
      #   # Loop over sub-trajectories
      #   for idx_sub_trajectory in np.split(np.arange(traj_length), p):
      #     # Get the input/target time indices
      #     idx_inp = idx_sub_trajectory[:1]
      #     idx_tgt = idx_sub_trajectory[:]
      #     # Split the dataset along the time axis
      #     u_inp = trajs[:, :, idx_inp]
      #     u_tgt = trajs[:, :, idx_tgt]
      #     # Get predictions and target
      #     u_prd = _predict_trajectory_autoregressively(
      #       state, specs, u_inp,
      #       (stats_trj_mean[:, idx_inp], stats_trj_std[:, idx_inp]),
      #       (stats_trj_mean[:, idx_tgt], stats_trj_std[:, idx_tgt]),
      #       idx_sub_trajectory.shape[0],
      #     )
      #     # Re-arrange the predictions gotten from each device
      #     u_prd = u_prd.reshape(FLAGS.batch_size, *u_prd.shape[2:])
      #     u_tgt = u_tgt.reshape(FLAGS.batch_size, *u_tgt.shape[2:])
      #     # Compute and store metrics
      #     error_ar_l1_per_var[p].append(rel_l1_error(u_prd, u_tgt))
      #     error_ar_l2_per_var[p].append(rel_l2_error(u_prd, u_tgt))

    # Aggregate over the batch dimension and compute norm per variable
    error_dr_l1 = jnp.median(jnp.concatenate(error_dr_l1), axis=0).item()
    error_dr_l2 = jnp.median(jnp.concatenate(error_dr_l2), axis=0).item()
    for p in parts:
      # TMP
      # error_l1_per_var_agg = jnp.median(jnp.concatenate(error_ar_l1_per_var[p]), axis=0)
      # error_l2_per_var_agg = jnp.median(jnp.concatenate(error_ar_l2_per_var[p]), axis=0)
      # error_ar_l1[p] = jnp.sqrt(jnp.sum(jnp.power(error_l1_per_var_agg[p], 2))).item()
      # error_ar_l2[p] = jnp.sqrt(jnp.sum(jnp.power(error_l2_per_var_agg[p], 2))).item()
      error_ar_l1[p] = 0
      error_ar_l2[p] = 0

    # Build the metrics object
    metrics = EvalMetrics(
      error_autoreg_l1=[(p, errors) for p, errors in error_ar_l1.items()],
      error_autoreg_l2=[(p, errors) for p, errors in error_ar_l2.items()],
      error_direct_l1=error_dr_l1,
      error_direct_l2=error_dr_l2,
    )

    return metrics

  # Set the evaluation partitions
  autoreg_evaluation_parts = [1, 2, 5]  # FIXME: Get as input
  assert all([(num_times // p) >= FLAGS.direct_steps for p in autoreg_evaluation_parts])
  # Evaluate before training
  metrics_trn = evaluate(
    state=state,
    batches=dataset.batches(mode='train', batch_size=FLAGS.batch_size),
    parts=autoreg_evaluation_parts,
  )
  metrics_val = evaluate(
    state=state,
    batches=dataset.batches(mode='valid', batch_size=FLAGS.batch_size),
    parts=autoreg_evaluation_parts,
  )

  # Report the initial evaluations
  time_tot_pre = time() - time_int_pre
  logging.info('\t'.join([
    f'EPCH: {0 : 04d}/{epochs : 04d}',
    f'TIME: {time_tot_pre : 06.1f}s',
    f'LR: {state.opt_state.hyperparams["learning_rate"].item() : .2e}',
    f'RMSE: {0. : .2e}',
    f'L2-AR: {np.mean(metrics_val.error_autoreg_l2[0][1]) * 100 : .2f}%',
    f'L2-DR: {metrics_val.error_direct_l2 * 100 : .2f}%',
  ]))

  # Set up the checkpoint manager
  with disable_logging(level=logging.FATAL):
    (DIR / 'metrics').mkdir()
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpointer_options = orbax.checkpoint.CheckpointManagerOptions(
      max_to_keep=1,
      keep_period=None,
      # best_fn=(lambda metrics: np.mean(metrics['valid']['autoreg']['l2'][1][1]).item()),
      best_mode='min',
      create=True,)
    checkpointer_save_args = orbax_utils.save_args_from_target(target={'state': state})
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
      (DIR / 'checkpoints'), checkpointer, checkpointer_options)

  for epoch in range(1, epochs+1):
    # Store the initial time
    time_int = time()

    # Train one epoch
    subkey_0, subkey_1, key = jax.random.split(key, num=3)
    state, loss, grad = train_one_epoch(
      state=state,
      batches=dataset.batches(mode='train', batch_size=FLAGS.batch_size, key=subkey_0),
      key=subkey_1
    )

    # Evaluate
    metrics_trn = evaluate(
      state=state,
      batches=dataset.batches(mode='train', batch_size=FLAGS.batch_size),
      parts=autoreg_evaluation_parts,
    )
    metrics_val = evaluate(
      state=state,
      batches=dataset.batches(mode='valid', batch_size=FLAGS.batch_size),
      parts=autoreg_evaluation_parts,
    )

    # Log the results
    time_tot = time() - time_int
    logging.info('\t'.join([
      f'EPCH: {epoch : 04d}/{epochs : 04d}',
      f'TIME: {time_tot : 06.1f}s',
      f'LR: {state.opt_state.hyperparams["learning_rate"].item() : .2e}',
      f'RMSE: {np.sqrt(loss).item() : .2e}',
      f'GRAD: {grad.item() : .2e}',
      f'L2-AR: {np.mean(metrics_val.error_autoreg_l2[0][1]) * 100 : .2f}%',
      f'L2-DR: {metrics_val.error_direct_l2 * 100 : .2f}%',
    ]))

    with disable_logging(level=logging.FATAL):
      checkpoint_metrics = {
        'loss': loss.item(),
        'train': {
          'autoreg': {
            'l1': metrics_trn.error_autoreg_l1,
            'l2': metrics_trn.error_autoreg_l2,
          },
          'direct': {
            'l1': metrics_trn.error_direct_l1,
            'l2': metrics_trn.error_direct_l2,
          },
        },
        'valid': {
          'autoreg': {
            'l1': metrics_val.error_autoreg_l1,
            'l2': metrics_val.error_autoreg_l2,
          },
          'direct': {
            'l1': metrics_val.error_direct_l1,
            'l2': metrics_val.error_direct_l2,
          },
        },
      }
      # Store the state and the metrics
      checkpoint_manager.save(
        step=epoch,
        items={'state': state,},
        metrics=checkpoint_metrics,
        save_kwargs={'save_args': checkpointer_save_args}
      )
      with open(DIR / 'metrics' / f'{str(epoch)}.json', 'w') as f:
        json.dump(checkpoint_metrics, f)

  return state

def get_model(model_configs: Mapping[str, Any]) -> AbstractOperator:

  model = GraphNeuralPDESolver(
    **model_configs,
  )

  return model

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Check the available devices
  with disable_logging():
    process_index = jax.process_index()
    process_count = jax.process_count()
    local_devices = jax.local_devices()
  logging.info('JAX host: %d / %d', process_index, process_count)
  logging.info('JAX local devices: %r', local_devices)
  # We only support single-host training.
  assert process_count == 1

  # Initialize the random key
  key = jax.random.PRNGKey(SEED)

  # Read the dataset
  key, subkey = jax.random.split(key)
  experiment = FLAGS.experiment
  dataset = Dataset(
    dir='/'.join([FLAGS.datadir, (experiment + '.nc')]),
    n_train=(32 if FLAGS.debug else 2**14),
    n_valid=(32 if FLAGS.debug else 1024),
    n_test=(32 if FLAGS.debug else 1024),
    key=subkey,
  )

  # Read the checkpoint
  if FLAGS.params:
    DIR_OLD_EXPERIMENT = DIR_EXPERIMENTS / FLAGS.params
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    step = orbax.checkpoint.CheckpointManager(DIR_OLD_EXPERIMENT / 'checkpoints', orbax_checkpointer).latest_step()
    ckpt = orbax_checkpointer.restore(directory=(DIR_OLD_EXPERIMENT / 'checkpoints' / str(step) / 'default'))
    state = ckpt['state']
    with open(DIR_OLD_EXPERIMENT / 'configs.json', 'rb') as f:
      model_kwargs = json.load(f)['model_configs']
  else:
    state = None
    model_kwargs = None

  # Get the model
  if not model_kwargs:
    model_kwargs = dict(
      num_outputs=dataset.sample[0].shape[-1],
      num_grid_nodes=dataset.sample[0].shape[2:4],
      num_mesh_nodes=(64, 64),  # TRY: tune
      overlap_factor=2.0,  # TRY: tune
      num_multimesh_levels=4,  # TRY: tune
      latent_size=FLAGS.latent_size,  # TRY: tune
      num_mlp_hidden_layers=2,  # TRY: 1, 2, 3
      num_message_passing_steps=6,  # TRY: tune
    )
    if FLAGS.debug:
      model_kwargs['num_mesh_nodes'] = (4,4)
      model_kwargs['overlap_factor'] = 1.0
      model_kwargs['num_multimesh_levels'] = 1
      model_kwargs['latent_size'] = 8
      model_kwargs['num_mlp_hidden_layers'] = 1
      model_kwargs['num_message_passing_steps'] = 2

  model = get_model(model_kwargs)

  # Store the configurations
  DIR.mkdir()
  logging.info(f'Experiment stored in {DIR.relative_to(DIR_EXPERIMENTS).as_posix()}')
  flags = {f: FLAGS.get_flag_value(f, default=None) for f in FLAGS}
  with open(DIR / 'configs.json', 'w') as f:
    json.dump(fp=f,
      obj={'flags': flags, 'model_configs': model.configs},
      indent=2,
    )

  # Train the model
  key, subkey = jax.random.split(key)
  state = train(
    model=model,
    dataset=dataset,
    epochs=FLAGS.epochs,
    key=subkey,
    params=(state['params'] if state else None),
  )

if __name__ == '__main__':
  logging.set_verbosity('info')
  app.run(main)
