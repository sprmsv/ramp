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
from graphneuralpdesolver.autoregressive import AutoregressivePredictor
from graphneuralpdesolver.dataset import read_datasets, shuffle_arrays, normalize, unnormalize, Dataset
from graphneuralpdesolver.models.graphneuralpdesolver import GraphNeuralPDESolver, AbstractOperator, ToyOperator
from graphneuralpdesolver.utils import disable_logging, Array
from graphneuralpdesolver.metrics import mse, rel_l2_error, rel_l1_error


SEED = 43

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

PDETYPE = {
  'E1': 'CE',
  'E2': 'CE',
  'E3': 'CE',
  'WE1': 'WE',
  'WE2': 'WE',
  'WE3': 'WE',
  'KF': 'KF',
  'RP': 'AD',
  'MSWG': 'AD',
}

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

  # Set constants
  num_samples_trn = dataset.nums['train']
  num_times = sample_traj.shape[1]
  num_grid_points = sample_traj.shape[2:4]
  num_vars = sample_traj.shape[-1]
  batch_size = FLAGS.batch_size
  unroll_offset = FLAGS.unroll_steps * FLAGS.direct_steps
  assert num_samples_trn % batch_size == 0

  # Store the initial time
  time_int_pre = time()

  # Set the normalization statistics
  # TRY: Get statistics along the time axis
  stats_trj_mean = dataset.mean_trn
  stats_trj_std = dataset.mean_trn

  # Initialzize the model or use the loaded parameters
  if params:
    variables = {'params': params}
  else:
    subkey, key = jax.random.split(key)
    model_init_kwargs = dict(
      u_inp=jnp.ones_like(sample_traj).repeat(batch_size, axis=0),
      ndt=1.,
      specs=(
        jnp.ones_like(sample_spec).repeat(batch_size, axis=0)
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
  num_batches = num_samples_trn // batch_size
  lead_times = jnp.arange(unroll_offset, num_times - FLAGS.direct_steps)
  num_lead_times = num_times - unroll_offset - FLAGS.direct_steps

  # Set up the optimization components
  criterion_loss = mse
  lr = optax.cosine_decay_schedule(
    init_value=FLAGS.lr,
    decay_steps=(FLAGS.epochs * num_batches * FLAGS.direct_steps * num_lead_times),
    alpha=FLAGS.lr_decay,
  ) if FLAGS.lr_decay else FLAGS.lr
  tx = optax.inject_hyperparams(optax.adamw)(learning_rate=lr, weight_decay=1e-8)
  state = TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)

  # Define the autoregressive predictor
  predictor = AutoregressivePredictor(operator=model, num_steps_direct=FLAGS.direct_steps)

  def compute_loss(params: flax.typing.Collection, specs: Array,
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
    u_prd = predictor._apply_operator(variables, specs=specs, u_inp=u_inp, ndt=ndt)

    return criterion_loss(u_prd, u_tgt)

  def get_noisy_input(params: flax.typing.Collection, specs: Array,
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

  def get_loss_and_grads(params: flax.typing.Collection, specs: Array,
    u_lag: Array, u_tgt: Array, ndt: int) -> Tuple[Array, PyTreeDef]:
    """
    Computes the loss and the gradients of the loss w.r.t the parameters.
    """

    # Split the unrolling steps randomly to cut the gradients along the way
    # MODIFY: Change to JAX-generated random number (reproducability)
    noise_steps = np.random.choice(FLAGS.unroll_steps+1)
    grads_steps = FLAGS.unroll_steps - noise_steps

    # Get noisy input
    u_inp = get_noisy_input(
      params, specs, u_lag, num_steps_autoreg=noise_steps)
    # Use noisy input and compute gradients
    loss, grads = jax.value_and_grad(compute_loss)(
      params, specs, u_inp, ndt, u_tgt, num_steps_autoreg=grads_steps)

    return loss, grads

  def get_loss_and_grads_sub_batch(
    state: TrainState, key: flax.typing.PRNGKey,
    specs: Array, u_lag: Array, u_tgt: Array, ndt: int,
  ) -> Tuple[Array, PyTreeDef]:
    # NOTE: INPUT SHAPES [batch_size * num_lead_times, ...]

    # Shuffle the input/outputs along the batch axis
    specs, u_lag, u_tgt = shuffle_arrays(key, [specs, u_lag, u_tgt])

    # Split into num_lead_times chunks and get loss and gradients
    # -> [num_lead_times, batch_size, ...]
    specs = jnp.stack(jnp.split(specs, num_lead_times))
    u_lag = jnp.stack(jnp.split(u_lag, num_lead_times))
    u_tgt = jnp.stack(jnp.split(u_tgt, num_lead_times))

    def _update_state_on_mini_batch(i, carry):
      _state, _loss_mean = carry
      _loss, _grads = get_loss_and_grads(
        params=_state.params,
        specs=specs[i],
        u_lag=u_lag[i],
        u_tgt=u_tgt[i],
        ndt=ndt,
      )
      # Update the state by applying the gradients
      _state = _state.apply_gradients(grads=_grads)
      _loss_mean += _loss / num_lead_times
      return _state, _loss_mean

    state, loss = jax.lax.fori_loop(
      lower=0,
      upper=num_lead_times,
      body_fun=_update_state_on_mini_batch,
      init_val=(state, 0.)
    )

    return state, loss

  @jax.jit
  def train_one_batch(
    state: TrainState, batch: Tuple[Array, Array],
    key: flax.typing.PRNGKey) -> Tuple[TrainState, Array]:
    """TODO: WRITE."""

    # Unwrap the batch and normalize
    trajs, specs = batch
    trajs = normalize(trajs, mean=stats_trj_mean, std=stats_trj_std)

    # Get input output pairs for all lead times
    # -> [num_lead_times, batch_size, ...]
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
    )

    # Concatenate lead times along the batch axis
    # -> [batch_size * num_lead_times, ...]
    u_lag_batch = u_lag_batch.reshape(
        (batch_size * num_lead_times), 1, *num_grid_points, -1)
    u_tgt_batch = u_tgt_batch.reshape(
        (batch_size * num_lead_times), FLAGS.direct_steps, *num_grid_points, -1)
    specs_batch = specs_batch.reshape(
        (batch_size * num_lead_times), -1)

    # Compute loss and gradient by mapping on the time axis
    # Same u_lag and specs, loop over ndt
    subkeys = jnp.stack(jax.random.split(key, num=FLAGS.direct_steps))
    ndt_batch = 1 + jnp.arange(FLAGS.direct_steps)  # -> [direct_steps,]
    u_tgt_batch = jnp.expand_dims(u_tgt_batch, axis=2).swapaxes(0, 1)  # -> [direct_steps, ...]

    def update_state_on_sub_batch(i, carry):
      _state, _loss_mean = carry
      _state, _loss = get_loss_and_grads_sub_batch(
        state=_state,
        key=subkeys[i],
        specs=specs_batch,
        u_lag=u_lag_batch,
        u_tgt=u_tgt_batch[i],
        ndt=ndt_batch[i],
      )
      _loss_mean += _loss / FLAGS.direct_steps
      return _state, _loss_mean

    state, loss = jax.lax.fori_loop(
      lower=0,
      upper=FLAGS.direct_steps,
      body_fun=update_state_on_sub_batch,
      init_val=(state, 0.)
    )

    return state, loss

  def train_one_epoch(state: TrainState, batches: Iterable[Tuple[Array, Array]],
    key: flax.typing.PRNGKey) -> Tuple[TrainState, Array]:
    """Updates the state based on accumulated losses and gradients."""

    # Loop over the batches
    loss_epoch = 0.
    for idx, batch in enumerate(batches):
      begin_batch = time()
      batch = jax.tree_map(jax.device_put, batch)  # Transfer to device memory
      assert jax.devices()[0] in batch[0].devices()  # TMP
      assert jax.devices()[0] in batch[1].devices()  # TMP
      subkey, key = jax.random.split(key)
      state, loss = train_one_batch(state, batch, subkey)
      loss_epoch += loss * batch_size / num_samples_trn
      time_batch = time() - begin_batch

      if FLAGS.verbose:
        logging.info('\t'.join([
          f'\t',
          f'BTCH: {idx+1:04d}/{num_batches:04d}',
          f'TIME: {time_batch:06.1f}s',
          f'RMSE: {np.sqrt(loss).item() : .2e}',
        ]))

    return state, loss_epoch

  @functools.partial(jax.jit, static_argnames=('num_steps',))
  def predict_trajectory(
      state: TrainState, specs: Array, u_inp: Array, num_steps: int,
      stats_inp: Tuple[Array, Array], stats_tgt: Tuple[Array, Array],
    ) -> Array:
    """
    Normalizes the input and predicts the trajectories autoregressively.
    The input dataset must be raw (not normalized).
    """

    # Normalize the input
    u_inp = normalize(u_inp, mean=stats_inp[0], std=stats_inp[1])
    # Get normalized predictions
    variables = {'params': state.params}
    rollout, _ = predictor.urnoll(
      variables=variables,
      specs=specs,
      u_inp=u_inp,
      num_steps=num_steps,
    )
    # Denormalize the predictions
    rollout = unnormalize(rollout, mean=stats_tgt[0], std=stats_tgt[1])

    return rollout

  @jax.jit
  def evaluate_direct_prediction(
      state: TrainState, batch: Tuple[Array, Array]) -> Tuple[Array, Array]:

    # Unwrap the batch
    trajs, specs = batch

    # Set lead times
    lead_times = jnp.arange(num_times - FLAGS.direct_steps)
    num_lead_times = num_times - FLAGS.direct_steps

    # Get input output pairs for all lead times
    # -> [num_lead_times, batch_size, ...]
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
    )
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
    mean_tgt = jax.vmap(
        lambda lt: jax.lax.dynamic_slice_in_dim(
          operand=stats_trj_mean,
          start_index=(lt+1), slice_size=FLAGS.direct_steps, axis=1)
    )(lead_times)
    std_tgt = jax.vmap(
        lambda lt: jax.lax.dynamic_slice_in_dim(
          operand=stats_trj_std,
          start_index=(lt+1), slice_size=FLAGS.direct_steps, axis=1)
    )(lead_times)

    def get_direct_errors(lt, carry):
      err_l1_mean, err_l2_mean = carry
      def get_direct_prediction(ndt, forcing):
        u_inp_nrm = normalize(u_inp[lt], mean=mean_inp[lt], std=std_inp[lt])
        u_prd_nrm = model.apply(
          variables={'params': state.params},
          u_inp=u_inp_nrm,
          specs=specs[lt],
          ndt=ndt,
        )
        u_prd = unnormalize(u_prd_nrm, mean=mean_tgt[lt][:, ndt-1], std=std_tgt[lt][:, ndt-1])
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
      init_val=(jnp.zeros(shape=(1,)), jnp.zeros(shape=(1,)))
    )

    return err_l1_mean, err_l2_mean

  def evaluate(state: TrainState, batches: Iterable[Tuple[Array, Array]],
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
        assert jax.devices()[0] in trajs.devices()  # TMP
        assert jax.devices()[0] in specs.devices()  # TMP

        # Evaluate direct prediction
        _error_dr_l1_batch, _error_dr_l2_batch = evaluate_direct_prediction(
          state=state,
          batch=batch,
        )
        error_dr_l1.append(_error_dr_l1_batch)
        error_dr_l2.append(_error_dr_l2_batch)

        # Evaluate autoregressive prediction
        for p in parts:
          # Get a dividable full trajectory
          traj_length = num_times - (num_times % p)
          # Loop over sub-trajectories
          for idx_sub_trajectory in np.split(np.arange(traj_length), p):
            # Get the input/target time indices
            idx_inp = idx_sub_trajectory[:1]
            idx_tgt = idx_sub_trajectory[:]
            # Split the dataset along the time axis
            u_inp = trajs[:, idx_inp]
            u_tgt = trajs[:, idx_tgt]
            # Get predictions and target
            pred = predict_trajectory(
              state=state,
              specs=specs,
              u_inp=u_inp,
              num_steps=u_tgt.shape[1],
              stats_inp=(stats_trj_mean[:, idx_inp], stats_trj_std[:, idx_inp]),
              stats_tgt=(stats_trj_mean[:, idx_tgt], stats_trj_std[:, idx_tgt]),
            )
            # Compute and store metrics
            error_ar_l1_per_var[p].append(rel_l1_error(pred, u_tgt))
            error_ar_l2_per_var[p].append(rel_l2_error(pred, u_tgt))

      # Aggregate over the batch dimension and compute norm per variable
      error_dr_l1 = jnp.median(jnp.concatenate(error_dr_l1), axis=0).item()
      error_dr_l2 = jnp.median(jnp.concatenate(error_dr_l2), axis=0).item()
      for p in parts:
        error_l1_per_var_agg = jnp.median(jnp.concatenate(error_ar_l1_per_var[p]), axis=0)
        error_l2_per_var_agg = jnp.median(jnp.concatenate(error_ar_l2_per_var[p]), axis=0)
        error_ar_l1[p] = jnp.sqrt(jnp.sum(jnp.power(error_l1_per_var_agg[p], 2))).item()
        error_ar_l2[p] = jnp.sqrt(jnp.sum(jnp.power(error_l2_per_var_agg[p], 2))).item()

      # Build the metrics object
      metrics = EvalMetrics(
        error_autoreg_l1=[(p, errors) for p, errors in error_ar_l1.items()],
        error_autoreg_l2=[(p, errors) for p, errors in error_ar_l2.items()],
        error_direct_l1=error_dr_l1,
        error_direct_l2=error_dr_l2,
      )

      return metrics

  # Set the evaluation partitions
  autoreg_evaluation_parts = [1, 2, 5]  # FIXME: Get as an input
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
    state, loss = train_one_epoch(
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

  # Read the dataset
  experiment = FLAGS.experiment
  dataset = Dataset(
    dir=(FLAGS.datadir + experiment + '.nc'),
    n_train=(2**14),
    n_valid=32,  # TMP
    n_test=1024,
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
      num_mesh_nodes=(4, 4),  # TMP
      overlap_factor=1.0,
      num_multimesh_levels=3,  # TMP
    )
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
  key = jax.random.PRNGKey(SEED)
  state = train(
    model=model,
    dataset=dataset,
    epochs=FLAGS.epochs,
    key=key,
    params=(state['params'] if state else None),
  )

if __name__ == '__main__':
  logging.set_verbosity('info')
  app.run(main)
