from datetime import datetime
import functools
from time import time
from typing import Tuple, Any, Mapping, Iterable, Callable
import json
import pickle

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
from flax.training.common_utils import shard, shard_prng_key
from flax.jax_utils import replicate, unreplicate
import orbax.checkpoint

from mpgno.experiments import DIR_EXPERIMENTS
from mpgno.autoregressive import AutoregressivePredictor, OperatorNormalizer
from mpgno.dataset import Dataset
from mpgno.models.mpgno import MPGNO, AbstractOperator
from mpgno.utils import disable_logging, Array, shuffle_arrays, split_arrays, normalize
from mpgno.metrics import mse_loss, rel_l1_loss
from mpgno.metrics import BatchMetrics, Metrics, EvalMetrics
from mpgno.metrics import mse_error, rel_l2_error, rel_l1_error
from mpgno.metrics import rel_l1_error_sum_vars, rel_l2_error_sum_vars


NUM_DEVICES = jax.local_device_count()

TIME_DOWNSAMPLE_FACTOR = 2
IDX_FN = 7
# NOTE: With JUMP_STEPS=4, we need trajectories of length 12 after downsampling
MAX_JUMP_STEPS = 1

EVAL_FREQ = 50

# FLAGS::general
FLAGS = flags.FLAGS
flags.DEFINE_string(name='exp', default='000', required=False,
  help='Name of the experiment'
)
flags.DEFINE_string(name='datetime', default=None, required=False,
  help='A string representing the current datetime'
)
flags.DEFINE_string(name='datadir', default=None, required=True,
  help='Path of the folder containing the datasets'
)
flags.DEFINE_string(name='datapath', default=None, required=True,
  help='Relative path inside the data directory'
)
flags.DEFINE_string(name='params', default=None, required=False,
  help='Path of the previous experiment containing the initial parameters'
)
flags.DEFINE_integer(name='seed', default=44, required=False,
  help='Seed for random number generator'
)

# FLAGS::training
flags.DEFINE_integer(name='batch_size', default=4, required=False,
  help='Size of a batch of training samples'
)
flags.DEFINE_integer(name='epochs', default=20, required=False,
  help='Number of training epochs'
)
flags.DEFINE_float(name='lr_init', default=1e-05, required=False,
  help='Initial learning rate in the onecycle scheduler'
)
flags.DEFINE_float(name='lr_peak', default=2e-04, required=False,
  help='Peak learning rate in the onecycle scheduler'
)
flags.DEFINE_float(name='lr_base', default=1e-05, required=False,
  help='Final learning rate in the onecycle scheduler'
)
flags.DEFINE_float(name='lr_lowr', default=1e-06, required=False,
  help='Final learning rate in the exponential decay'
)
flags.DEFINE_integer(name='jump_steps', default=1, required=False,
  help='Factor by which the dataset time delta is multiplied in prediction'
)
flags.DEFINE_integer(name='direct_steps', default=1, required=False,
  help='Maximum number of time steps between input/output pairs during training'
)
flags.DEFINE_integer(name='unroll_steps', default=0, required=False,
  help='Number of steps for getting a noisy input and applying the model autoregressively'
)
flags.DEFINE_integer(name='n_train', default=(2**9), required=False,
  help='Number of training samples'
)
flags.DEFINE_integer(name='n_valid', default=(2**8), required=False,
  help='Number of validation samples'
)

# FLAGS::model
flags.DEFINE_integer(name='num_mesh_nodes', default=64, required=False,
  help='Number of mesh nodes in each dimension'
)
flags.DEFINE_integer(name='deriv_degree', default=2, required=False,
  help='Maximum degree of auxiliary partial derivatives'
)
flags.DEFINE_float(name='overlap_factor_grid2mesh', default=2.0, required=False,
  help='Overlap factor for grid2mesh edges (encoder)'
)
flags.DEFINE_float(name='overlap_factor_mesh2grid', default=2.0, required=False,
  help='Overlap factor for mesh2grid edges (decoder)'
)
flags.DEFINE_integer(name='num_multimesh_levels', default=4, required=False,
  help='Number of multimesh connection levels (processor)'
)
flags.DEFINE_integer(name='node_coordinate_freqs', default=2, required=False,
  help='Number of frequencies for encoding periodic node coordinates'
)
flags.DEFINE_integer(name='latent_size', default=128, required=False,
  help='Size of latent node and edge features'
)
flags.DEFINE_integer(name='num_mlp_hidden_layers', default=1, required=False,
  help='Number of hidden layers of all MLPs'
)
flags.DEFINE_integer(name='num_message_passing_steps', default=18, required=False,
  help='Number of message-passing steps in the processor'
)
flags.DEFINE_integer(name='num_message_passing_steps_grid', default=2, required=False,
  help='Number of message-passing steps in the decoder'
)
flags.DEFINE_float(name='p_dropout_edges_grid2mesh', default=0.5, required=False,
  help='Probability of dropping out edges of grid2mesh'
)
flags.DEFINE_float(name='p_dropout_edges_mesh2grid', default=0., required=False,
  help='Probability of dropping out edges of mesh2grid'
)

def train(
  key: flax.typing.PRNGKey,
  model: nn.Module,
  state: TrainState,
  dataset: Dataset,
  jump_steps: int,
  direct_steps: int,
  unroll_steps: int,
  epochs: int,
  epochs_before: int = 0,
  loss_fn: Callable = rel_l1_loss,
) -> TrainState:
  """Trains a model and returns the state."""

  # Samples
  sample_traj, sample_spec = dataset.sample
  _use_specs = (sample_spec is not None)
  sample_traj = jnp.array(sample_traj)
  sample_spec = jnp.array(sample_spec) if _use_specs else None

  # Set constants
  num_samples_trn = dataset.nums['train']
  len_traj = dataset.shape[1]
  num_grid_points = dataset.shape[2:4]
  num_vars = dataset.shape[-1]
  unroll_offset = unroll_steps * direct_steps
  assert num_samples_trn % FLAGS.batch_size == 0
  num_batches = num_samples_trn // FLAGS.batch_size
  assert (jump_steps * FLAGS.batch_size) % NUM_DEVICES == 0
  batch_size_per_device = (jump_steps * FLAGS.batch_size) // NUM_DEVICES
  assert len_traj % jump_steps == 0
  num_times = len_traj // jump_steps
  evaluation_frequency = (
    (FLAGS.epochs // EVAL_FREQ) if (FLAGS.epochs >= EVAL_FREQ)
    else 1
  )

  # Store the initial time
  time_int_pre = time()

  # Define the permissible lead times
  num_lead_times = num_times - 1 - unroll_offset
  assert num_lead_times > 0
  assert unroll_offset + direct_steps < num_times
  num_lead_times_full = max(0, num_times - direct_steps - unroll_offset)
  num_lead_times_part = num_lead_times - num_lead_times_full
  num_valid_pairs = (
    num_lead_times_full * direct_steps
    + (num_lead_times_part * (num_lead_times_part+1) // 2)
  )
  lead_times = jnp.arange(unroll_offset, num_times - 1)

  # Define the autoregressive predictor
  normalizer = OperatorNormalizer(operator=model)
  predictor = AutoregressivePredictor(
    normalizer=normalizer,
    num_steps_direct=direct_steps,
    tau_base=jump_steps
  )

  # Set the normalization statistics
  stats = {
    key: {
      k: (jnp.array(v) if v is not None else None)
      for k, v in val.items()
    }
    for key, val in dataset.stats.items()
  }

  # Replicate state and stats
  # NOTE: Internally uses jax.device_put_replicate
  state = replicate(state)
  stats = replicate(stats)

  @functools.partial(jax.pmap, axis_name='device')
  def _train_one_batch(
    key: flax.typing.PRNGKey,
    state: TrainState,
    stats: dict,
    trajs: Array,
    times: Array,
    specs: Array,
  ) -> Tuple[TrainState, Array, Array]:
    """Loads a batch, normalizes it, updates the state based on it, and returns it."""

    def _update_state_per_subbatch(
      key: flax.typing.PRNGKey,
      state: TrainState,
      specs: Array,
      u_lag: Array,
      t_lag: Array,
      u_tgt: Array,
      tau: Array,
    ) -> Tuple[TrainState, Array, PyTreeDef]:
      # NOTE: INPUT SHAPES [batch_size_per_device, ...]

      def _get_loss_and_grads(
        key: flax.typing.PRNGKey,
        params: flax.typing.Collection,
        specs: Array,
        u_lag: Array,
        t_lag: Array,
        u_tgt: Array,
        tau: int,
      ) -> Tuple[Array, PyTreeDef]:
        """
        Computes the loss and the gradients of the loss w.r.t the parameters.
        """

        def _compute_loss(
          params: flax.typing.Collection,
          specs: Array,
          u_lag: Array,
          t_lag: Array,
          tau: int,
          u_tgt: Array,
          num_steps_autoreg: int,
          key: flax.typing.PRNGKey,
        ) -> Array:
          """Computes the prediction of the model and returns its loss."""

          variables = {'params': params}
          # Apply autoregressive steps
          key, subkey = jax.random.split(key)
          u_inp = predictor.jump(
            variables=variables,
            stats=stats,
            specs=specs,
            u_inp=u_lag,
            t_inp=t_lag,
            num_jumps=num_steps_autoreg,
            key=subkey,
          )
          t_inp = t_lag + num_steps_autoreg * jump_steps * direct_steps

          # Get the output
          key, subkey = jax.random.split(key)
          _loss_inputs = normalizer.get_loss_inputs(
            variables=variables,
            stats=stats,
            specs=specs,
            u_inp=u_inp,
            t_inp=t_inp,
            u_tgt=u_tgt,
            tau=tau,
            key=subkey,
          )

          return loss_fn(*_loss_inputs)

        def _get_noisy_input(
          specs: Array,
          u_lag: Array,
          t_lag: Array,
          num_steps_autoreg: int,
          key: flax.typing.PRNGKey,
        ) -> Array:
          """Apply the model to the lagged input to get a noisy input."""

          variables = {'params': params}
          u_inp_noisy = predictor.jump(
            variables=variables,
            stats=stats,
            specs=specs,
            u_inp=u_lag,
            t_inp=t_lag,
            num_jumps=num_steps_autoreg,
            key=key,
          )

          return u_inp_noisy

        # Split the unrolling steps randomly to cut the gradients along the way
        noise_steps = 0  # NOTE: Makes the training unstable
        grads_steps = unroll_steps - noise_steps

        # Get noisy input
        key, subkey = jax.random.split(key)
        u_inp = _get_noisy_input(
          specs, u_lag, t_lag, num_steps_autoreg=noise_steps, key=subkey)
        t_inp = t_lag + noise_steps * jump_steps * direct_steps
        # Use noisy input and compute gradients
        key, subkey = jax.random.split(key)
        loss, grads = jax.value_and_grad(_compute_loss)(
          params, specs, u_inp, t_inp, tau, u_tgt, num_steps_autoreg=grads_steps, key=subkey)

        return loss, grads

      # Update state, loss, and gradients
      _loss, _grads = _get_loss_and_grads(
        key=key,
        params=state.params,
        specs=(specs if _use_specs else None),
        u_lag=u_lag,
        t_lag=t_lag,
        u_tgt=u_tgt,
        tau=tau,
      )
      # Synchronize loss and gradients
      loss = jax.lax.pmean(_loss, axis_name='device')
      grads = jax.lax.pmean(_grads, axis_name='device')
      # Apply gradients
      state = state.apply_gradients(grads=grads)

      return state, loss, grads

    # Index trajectories and times and collect input/output pairs
    # -> [num_lead_times, batch_size_per_device, ...]
    u_lag_batch = jax.vmap(
        lambda lt: jax.lax.dynamic_slice_in_dim(
          operand=trajs,
          start_index=(lt-unroll_offset), slice_size=1, axis=1)
    )(lead_times)
    t_lag_batch = jax.vmap(
        lambda lt: jax.lax.dynamic_slice_in_dim(
          operand=times,
          start_index=(lt-unroll_offset), slice_size=1, axis=1)
    )(lead_times)
    specs_batch = (specs[None, :, None, :]
      .repeat(repeats=num_lead_times, axis=0)
    ) if _use_specs else None
    u_tgt_batch = jax.vmap(
        lambda lt: jax.lax.dynamic_slice_in_dim(
          operand=jnp.concatenate([trajs, jnp.zeros_like(trajs)], axis=1),
          start_index=(lt+1), slice_size=direct_steps, axis=1)
    )(lead_times)

    # Repeat inputs along the time axis to match with u_tgt
    # -> [num_lead_times, batch_size_per_device, direct_steps, ...]
    u_lag_batch = jnp.tile(u_lag_batch, reps=(1, 1, direct_steps, 1, 1, 1))
    t_lag_batch = jnp.tile(t_lag_batch, reps=(1, 1, direct_steps, 1, 1, 1))
    tau_batch = jnp.tile(
      (jump_steps * jnp.arange(1, direct_steps+1)).reshape(1, 1, direct_steps, 1),
      reps=(num_lead_times, batch_size_per_device, 1, 1)
    )
    specs_batch = jnp.tile(specs_batch, reps=(1, 1, direct_steps, 1)) if _use_specs else None

    # Put all pairs along the batch axis
    # -> [batch_size_per_device * num_lead_times * direct_steps, ...]
    u_lag_batch = u_lag_batch.reshape((num_lead_times*batch_size_per_device*direct_steps), 1, *num_grid_points, num_vars)
    t_lag_batch = t_lag_batch.reshape((num_lead_times*batch_size_per_device*direct_steps), 1)
    tau_batch = tau_batch.reshape((num_lead_times*batch_size_per_device*direct_steps), 1)
    specs_batch = specs_batch.reshape((num_lead_times*batch_size_per_device*direct_steps), -1) if _use_specs else None
    u_tgt_batch = u_tgt_batch.reshape((num_lead_times*batch_size_per_device*direct_steps), 1, *num_grid_points, num_vars)

    # Remove the invalid pairs
    # -> [batch_size_per_device * num_valid_pairs, ...]
    offset_full_lead_times = (num_times - direct_steps - unroll_offset) * direct_steps * batch_size_per_device
    idx_invalid_pairs = np.array([
      (offset_full_lead_times + (_d * batch_size_per_device + _b) * direct_steps - (_n + 1))
      for _d in range(direct_steps - 1)
      for _b in range(1, batch_size_per_device + 1)
      for _n in range(_d + 1)
    ]).astype(int)
    u_lag_batch = jnp.delete(u_lag_batch, idx_invalid_pairs, axis=0)
    t_lag_batch = jnp.delete(t_lag_batch, idx_invalid_pairs, axis=0)
    tau_batch = jnp.delete(tau_batch, idx_invalid_pairs, axis=0)
    specs_batch = jnp.delete(specs_batch, idx_invalid_pairs, axis=0) if _use_specs else None
    u_tgt_batch = jnp.delete(u_tgt_batch, idx_invalid_pairs, axis=0)

    # Shuffle and split the pairs
    # -> [num_valid_pairs, batch_size_per_device, ...]
    num_valid_pairs = u_tgt_batch.shape[0] // batch_size_per_device
    key, subkey = jax.random.split(key)
    if _use_specs:
      u_lag_batch, t_lag_batch, tau_batch, specs_batch, u_tgt_batch = shuffle_arrays(
        subkey, [u_lag_batch, t_lag_batch, tau_batch, specs_batch, u_tgt_batch])
      u_lag_batch, t_lag_batch, tau_batch, specs_batch, u_tgt_batch = split_arrays(
        [u_lag_batch, t_lag_batch, tau_batch, specs_batch, u_tgt_batch], size=batch_size_per_device)
    else:
      u_lag_batch, t_lag_batch, tau_batch, u_tgt_batch = shuffle_arrays(
        subkey, [u_lag_batch, t_lag_batch, tau_batch, u_tgt_batch])
      u_lag_batch, t_lag_batch, tau_batch, u_tgt_batch = split_arrays(
        [u_lag_batch, t_lag_batch, tau_batch, u_tgt_batch], size=batch_size_per_device)

    # Add loss and gradients for each subbatch
    def _update_state(i, carry):
      # Update state, loss, and gradients
      _state, _loss_carried, _grads_carried, _key_carried = carry
      _key_updated, _subkey = jax.random.split(_key_carried)
      _state, _loss_subbatch, _grads_subbatch = _update_state_per_subbatch(
        key=_subkey,
        state=_state,
        specs=(specs_batch if _use_specs else None),
        u_lag=u_lag_batch[i],
        t_lag=t_lag_batch[i],
        u_tgt=u_tgt_batch[i],
        tau=tau_batch[i],
      )
      # Update the carried loss and gradients of the subbatch
      _loss_updated = _loss_carried + _loss_subbatch / num_valid_pairs
      _grads_updated = jax.tree_map(
        lambda g_old, g_new: (g_old + g_new / num_valid_pairs),
        _grads_carried,
        _grads_subbatch,
      )

      return _state, _loss_updated, _grads_updated, _key_updated

    # Loop over the pairs
    _init_state = state
    _init_loss = 0.
    _init_grads = jax.tree_map(lambda p: jnp.zeros_like(p), state.params)
    key, _init_key = jax.random.split(key)
    state, loss, grads, _ = jax.lax.fori_loop(
      lower=0,
      upper=num_valid_pairs,
      body_fun=_update_state,
      init_val=(_init_state, _init_loss, _init_grads, _init_key)
    )

    # Synchronize loss and gradients
    # NOTE: Redundent since they are synchronized everytime before being applied
    # loss = jax.lax.pmean(loss, axis_name='device')
    # grads = jax.lax.pmean(grads, axis_name='device')

    return state, loss, grads

  def train_one_epoch(
    key: flax.typing.PRNGKey,
    state: TrainState,
    batches: Iterable[Tuple[Array, Array]],
  ) -> Tuple[TrainState, Array, Array]:
    """Updates the state based on accumulated losses and gradients."""

    # Loop over the batches
    loss_epoch = 0.
    grad_epoch = 0.
    for batch in batches:
      # Unwrap the batch
      # -> [batch_size, len_traj, ...]
      trajs, specs = batch
      times = jnp.tile(jnp.arange(trajs.shape[1]), reps=(trajs.shape[0], 1))

      # Downsample the trajectories
      # -> [batch_size * jump_steps, num_times, ...]
      trajs = jnp.concatenate(jnp.split(
          (trajs
          .reshape(FLAGS.batch_size, num_times, jump_steps, *num_grid_points, num_vars)
          .swapaxes(1, 2)
          .reshape(FLAGS.batch_size, len_traj, *num_grid_points, num_vars)),
          jump_steps,
          axis=1),
        axis=0,
      )
      times = jnp.concatenate(jnp.split(
          (times
          .reshape(FLAGS.batch_size, num_times, jump_steps)
          .swapaxes(1, 2)
          .reshape(FLAGS.batch_size, len_traj)),
          jump_steps,
          axis=1),
        axis=0,
      )
      specs = (jnp.tile(specs, jump_steps)
        .reshape(FLAGS.batch_size, jump_steps, -1)
        .swapaxes(0, 1)
        .reshape(FLAGS.batch_size * jump_steps, -1)
      ) if _use_specs else None

      # Split the batch between devices
      # -> [NUM_DEVICES, batch_size_per_device, ...]
      trajs = shard(trajs)
      times = shard(times)
      specs = shard(specs) if _use_specs else None

      # Get loss and updated state
      subkey, key = jax.random.split(key)
      subkey = shard_prng_key(subkey)
      state, loss, grads = _train_one_batch(subkey, state, stats, trajs, times, specs)
      # NOTE: Using the first element of replicated loss and grads
      loss_epoch += loss[0] * FLAGS.batch_size / num_samples_trn
      grad_epoch += np.mean(jax.tree_util.tree_flatten(
        jax.tree_map(jnp.mean, jax.tree_map(lambda g: jnp.abs(g[0]), grads)))[0]) / num_batches

    return state, loss_epoch, grad_epoch

  @jax.pmap
  def _evaluate_direct_prediction(
    state: TrainState,
    stats,
    trajs: Array,
    times: Array,
    specs: Array,
  ) -> Mapping:

    # Inputs are of shape [batch_size_per_device, ...]

    # Set lead times
    num_lead_times = num_times - direct_steps
    lead_times = jnp.arange(num_times - direct_steps)

    # Get input output pairs for all lead times
    # -> [num_lead_times, batch_size_per_device, ...]
    u_inp = jax.vmap(
        lambda lt: jax.lax.dynamic_slice_in_dim(
          operand=trajs,
          start_index=(lt), slice_size=1, axis=1)
    )(lead_times)
    t_inp = jax.vmap(
        lambda lt: jax.lax.dynamic_slice_in_dim(
          operand=times,
          start_index=(lt), slice_size=1, axis=1)
    )(lead_times)
    u_tgt = jax.vmap(
        lambda lt: jax.lax.dynamic_slice_in_dim(
          operand=trajs,
          start_index=(lt+1), slice_size=direct_steps, axis=1)
    )(lead_times)
    specs = (jnp.array(specs[None, :, :])
      .repeat(repeats=num_lead_times, axis=0)
    ) if _use_specs else None

    def get_direct_errors(lt, carry):
      carry = BatchMetrics(**carry)
      def get_direct_prediction(tau, forcing):
        u_prd = normalizer.apply(
          variables={'params': state.params},
          stats=stats,
          specs=(specs[lt] if _use_specs else None),
          u_inp=u_inp[lt],
          t_inp=t_inp[lt],
          tau=tau,
        )
        return (tau+jump_steps), u_prd
      _, u_prd = jax.lax.scan(
        f=get_direct_prediction,
        init=jump_steps,
        xs=None,
        length=direct_steps,
      )
      u_prd = u_prd.squeeze(axis=2).swapaxes(0, 1)

      # Get target variables (velocities) and normalize using global statistics
      _vel_prd = normalize(
        arr=u_prd[..., dataset.metadata.target_variables],
        shift=dataset.metadata.stats_target_variables['mean'],
        scale=dataset.metadata.stats_target_variables['std'],
      )
      _vel_tgt = normalize(
        arr=u_tgt[lt][..., dataset.metadata.target_variables],
        shift=dataset.metadata.stats_target_variables['mean'],
        scale=dataset.metadata.stats_target_variables['std'],
      )

      # Compute metrics
      batch_metrics = BatchMetrics(
        mse=(jnp.linalg.norm(mse_error(u_prd, u_tgt[lt]), axis=1) / num_lead_times),
        l1=(jnp.linalg.norm(rel_l1_error(u_prd, u_tgt[lt]), axis=1) / num_lead_times),
        l2=(jnp.linalg.norm(rel_l2_error(u_prd, u_tgt[lt]), axis=1) / num_lead_times),
        l1_alt=(rel_l1_error_sum_vars(_vel_prd, _vel_tgt) / num_lead_times),
        l2_alt=(rel_l2_error_sum_vars(_vel_prd, _vel_tgt) / num_lead_times),
      )

      carry += batch_metrics

      return carry.__dict__

    # Get mean errors per each sample in the batch
    init_metrics = BatchMetrics(
      mse=jnp.zeros(shape=(batch_size_per_device,)),
      l1=jnp.zeros(shape=(batch_size_per_device,)),
      l2=jnp.zeros(shape=(batch_size_per_device,)),
      l1_alt=jnp.zeros(shape=(batch_size_per_device,)),
      l2_alt=jnp.zeros(shape=(batch_size_per_device,)),
    )
    batch_metrics_mean = jax.lax.fori_loop(
      body_fun=get_direct_errors,
      lower=0,
      upper=num_lead_times,
      init_val=init_metrics.__dict__,
    )

    return batch_metrics_mean

  @jax.pmap
  def _evaluate_rollout_prediction(
    state: TrainState,
    stats,
    trajs: Array,
    times: Array,
    specs: Array,
  ) -> Mapping:
    """
    Predicts the trajectories autoregressively.
    The input dataset must be raw (not normalized).
    Inputs are of shape [batch_size_per_device, ...]
    """

    # Set input and target
    u_inp = trajs[:, :1]
    t_inp = times[:, :1]
    u_tgt = trajs

    # Get unrolled predictions
    variables = {'params': state.params}
    u_prd, _ = predictor.unroll(
      variables=variables,
      stats=stats,
      specs=specs,
      u_inp=u_inp,
      t_inp=t_inp,
      num_steps=num_times,
    )

    # Get target variables (velocities) and normalize using global statistics
    _vel_prd = normalize(
      arr=u_prd[..., dataset.metadata.target_variables],
      shift=dataset.metadata.stats_target_variables['mean'],
      scale=dataset.metadata.stats_target_variables['std'],
    )
    _vel_tgt = normalize(
      arr=u_tgt[..., dataset.metadata.target_variables],
      shift=dataset.metadata.stats_target_variables['mean'],
      scale=dataset.metadata.stats_target_variables['std'],
    )

    # Calculate the errors
    batch_metrics = BatchMetrics(
      mse=jnp.linalg.norm(mse_error(u_prd, u_tgt), axis=1),
      l1=jnp.linalg.norm(rel_l1_error(u_prd, u_tgt), axis=1),
      l2=jnp.linalg.norm(rel_l2_error(u_prd, u_tgt), axis=1),
      l1_alt=rel_l1_error_sum_vars(_vel_prd, _vel_tgt),
      l2_alt=rel_l2_error_sum_vars(_vel_prd, _vel_tgt),
    )

    return batch_metrics.__dict__

  @jax.pmap
  def _evaluate_final_prediction(
    state: TrainState,
    stats,
    trajs: Array,
    times: Array,
    specs: Array,
  ) -> Mapping:

    # Set input and target
    u_inp = trajs[:, :1]
    t_inp = times[:, :1]
    u_tgt = trajs[:, (IDX_FN):(IDX_FN+1)]

    # Get prediction at the final step
    _num_jumps = IDX_FN // (direct_steps * jump_steps)
    _num_direct_steps = IDX_FN % (direct_steps * jump_steps)
    variables = {'params': state.params}
    u_prd = predictor.jump(
      variables=variables,
      stats=stats,
      specs=specs,
      u_inp=u_inp,
      t_inp=t_inp,
      num_jumps=_num_jumps,
    )
    if _num_direct_steps:
      _, u_prd = predictor.unroll(
        variables=variables,
        stats=stats,
        specs=specs,
        u_inp=u_prd,
        t_inp=t_inp,
        num_steps=_num_direct_steps,
      )

    # Get target variables (velocities) and normalize using global statistics
    _vel_prd = normalize(
      arr=u_prd[..., dataset.metadata.target_variables],
      shift=dataset.metadata.stats_target_variables['mean'],
      scale=dataset.metadata.stats_target_variables['std'],
    )
    _vel_tgt = normalize(
      arr=u_tgt[..., dataset.metadata.target_variables],
      shift=dataset.metadata.stats_target_variables['mean'],
      scale=dataset.metadata.stats_target_variables['std'],
    )

    # Calculate the errors
    batch_metrics = BatchMetrics(
      mse=jnp.linalg.norm(mse_error(u_prd, u_tgt), axis=1),
      l1=jnp.linalg.norm(rel_l1_error(u_prd, u_tgt), axis=1),
      l2=jnp.linalg.norm(rel_l2_error(u_prd, u_tgt), axis=1),
      l1_alt=rel_l1_error_sum_vars(_vel_prd, _vel_tgt),
      l2_alt=rel_l2_error_sum_vars(_vel_prd, _vel_tgt),
    )

    return batch_metrics.__dict__

  def evaluate(
    state: TrainState,
    batches: Iterable[Tuple[Array, Array]],
    direct: bool = True,
    rollout: bool = False,
    final: bool = True,
  ) -> EvalMetrics:
    """Evaluates the model on a dataset based on multiple trajectory lengths."""

    metrics_direct: list[BatchMetrics] = []
    metrics_rollout: list[BatchMetrics] = []
    metrics_final: list[BatchMetrics] = []

    for batch in batches:
      # Unwrap the batch
      trajs_raw, specs_raw = batch
      times_raw = jnp.tile(jnp.arange(trajs_raw.shape[1]), reps=(trajs_raw.shape[0], 1))

      # Downsample the trajectories
      # -> [batch_size * jump_steps, num_times, ...]
      # NOTE: The last timestep is excluded to make the length of all the trajectories even
      trajs = jnp.concatenate(jnp.split(
          (trajs_raw
          .reshape(FLAGS.batch_size, num_times, jump_steps, *num_grid_points, num_vars)
          .swapaxes(1, 2)
          .reshape(FLAGS.batch_size, len_traj, *num_grid_points, num_vars)),
          jump_steps,
          axis=1),
        axis=0,
      )
      times = jnp.concatenate(jnp.split(
          (times_raw
          .reshape(FLAGS.batch_size, num_times, jump_steps)
          .swapaxes(1, 2)
          .reshape(FLAGS.batch_size, len_traj)),
          jump_steps,
          axis=1),
        axis=0,
      )
      specs = (jnp.tile(specs_raw, jump_steps)
        .reshape(FLAGS.batch_size, jump_steps, -1)
        .swapaxes(0, 1)
        .reshape(FLAGS.batch_size * jump_steps, -1)
      ) if _use_specs else None

      # Split the batch between devices
      # -> [NUM_DEVICES, batch_size_per_device, ...]
      trajs = shard(trajs)
      times = shard(times)
      specs = shard(specs) if _use_specs else None

      # Evaluate direct prediction
      if direct:
        batch_metrics_direct = _evaluate_direct_prediction(
          state, stats, trajs, times, specs,
        )
        batch_metrics_direct = BatchMetrics(**batch_metrics_direct)
        # Re-arrange the sub-batches gotten from each device
        batch_metrics_direct.reshape(shape=(batch_size_per_device * NUM_DEVICES, 1))
        # Append the metrics to the list
        metrics_direct.append(batch_metrics_direct)

      # Evaluate rollout prediction
      if rollout:
        batch_metrics_rollout = _evaluate_rollout_prediction(
          state, stats, trajs, times, specs
        )
        batch_metrics_rollout = BatchMetrics(**batch_metrics_rollout)
        # Re-arrange the sub-batches gotten from each device
        batch_metrics_rollout.reshape(shape=(batch_size_per_device * NUM_DEVICES, 1))
        # Compute and store metrics
        metrics_rollout.append(batch_metrics_rollout)

      # Split the batch between devices
      # -> [NUM_DEVICES, batch_size/NUM_DEVICES, ...]
      trajs = shard(trajs_raw)
      times = shard(times_raw)
      specs = shard(specs_raw) if _use_specs else None

      # Evaluate final prediction
      if final:
        assert IDX_FN < trajs.shape[2]
        batch_metrics_final = _evaluate_final_prediction(
          state, stats, trajs, times, specs,
        )
        batch_metrics_final = BatchMetrics(**batch_metrics_final)
        # Re-arrange the sub-batches gotten from each device
        batch_metrics_final.reshape(shape=(batch_size_per_device * (NUM_DEVICES // jump_steps), 1))
        # Append the errors to the list
        metrics_final.append(batch_metrics_final)

    # Aggregate over the batch dimension and compute norm per variable
    metrics_direct = Metrics(
      mse=jnp.median(jnp.concatenate([m.mse for m in metrics_direct]), axis=0).item(),
      l1=jnp.median(jnp.concatenate([m.l1 for m in metrics_direct]), axis=0).item(),
      l2=jnp.median(jnp.concatenate([m.l2 for m in metrics_direct]), axis=0).item(),
      l1_alt=jnp.median(jnp.concatenate([m.l1_alt for m in metrics_direct]), axis=0).item(),
      l2_alt=jnp.median(jnp.concatenate([m.l2_alt for m in metrics_direct]), axis=0).item(),
    ) if direct else None
    metrics_rollout = Metrics(
      mse=jnp.median(jnp.concatenate([m.mse for m in metrics_rollout]), axis=0).item(),
      l1=jnp.median(jnp.concatenate([m.l1 for m in metrics_rollout]), axis=0).item(),
      l2=jnp.median(jnp.concatenate([m.l2 for m in metrics_rollout]), axis=0).item(),
      l1_alt=jnp.median(jnp.concatenate([m.l1_alt for m in metrics_rollout]), axis=0).item(),
      l2_alt=jnp.median(jnp.concatenate([m.l2_alt for m in metrics_rollout]), axis=0).item(),
    ) if rollout else None
    metrics_final = Metrics(
      mse=jnp.median(jnp.concatenate([m.mse for m in metrics_final]), axis=0).item(),
      l1=jnp.median(jnp.concatenate([m.l1 for m in metrics_final]), axis=0).item(),
      l2=jnp.median(jnp.concatenate([m.l2 for m in metrics_final]), axis=0).item(),
      l1_alt=jnp.median(jnp.concatenate([m.l1_alt for m in metrics_final]), axis=0).item(),
      l2_alt=jnp.median(jnp.concatenate([m.l2_alt for m in metrics_final]), axis=0).item(),
    ) if final else None

    # Build the metrics object
    metrics = EvalMetrics(
      direct=(metrics_direct if direct else Metrics()),
      rollout=(metrics_rollout if rollout else Metrics()),
      final=(metrics_final if final else Metrics()),
    )

    return metrics

  # Evaluate before training
  metrics_trn = evaluate(
    state=state,
    batches=dataset.batches(mode='train', batch_size=FLAGS.batch_size),
  )
  metrics_val = evaluate(
    state=state,
    batches=dataset.batches(mode='valid', batch_size=FLAGS.batch_size),
  )

  # Report the initial evaluations
  time_tot_pre = time() - time_int_pre
  logging.info('\t'.join([
    f'DRCT: {direct_steps : 02d}',
    f'URLL: {unroll_steps : 02d}',
    f'EPCH: {epochs_before : 04d}/{FLAGS.epochs : 04d}',
    f'LR: {state.opt_state[-1].hyperparams["learning_rate"][0].item() : .2e}',
    f'TIME: {time_tot_pre : 06.1f}s',
    f'GRAD: {0. : .2e}',
    f'RMSE: {0. : .2e}',
    f'L2-DR-TRN: {metrics_trn.direct.l2 * 100 : .2f}%',
    f'L1-DR: {metrics_val.direct.l1 * 100 : .2f}%',
    f'L2-DR: {metrics_val.direct.l2 * 100 : .2f}%',
    f'L1-FN: {metrics_val.final.l1 * 100 : .2f}%',
    f'L2-FN: {metrics_val.final.l2 * 100 : .2f}%',
  ]))

  # Set up the checkpoint manager
  DIR = DIR_EXPERIMENTS / f'E{FLAGS.exp}' / FLAGS.datapath / FLAGS.datetime
  with disable_logging(level=logging.FATAL):
    (DIR / 'metrics').mkdir(exist_ok=True)
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpointer_options = orbax.checkpoint.CheckpointManagerOptions(
      max_to_keep=1,
      keep_period=None,
      best_fn=(lambda metrics: metrics['valid']['direct']['l2']),
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
      key=subkey_1,
      state=state,
      batches=dataset.batches(mode='train', batch_size=FLAGS.batch_size, key=subkey_0),
    )

    if (epoch % evaluation_frequency) == 0:
      # Evaluate
      metrics_trn = evaluate(
        state=state,
        batches=dataset.batches(mode='train', batch_size=FLAGS.batch_size),
      )
      metrics_val = evaluate(
        state=state,
        batches=dataset.batches(mode='valid', batch_size=FLAGS.batch_size),
      )

      # Log the results
      time_tot = time() - time_int
      logging.info('\t'.join([
        f'DRCT: {direct_steps : 02d}',
        f'URLL: {unroll_steps : 02d}',
        f'EPCH: {epochs_before + epoch : 04d}/{FLAGS.epochs : 04d}',
        f'LR: {state.opt_state[-1].hyperparams["learning_rate"][0].item() : .2e}',
        f'TIME: {time_tot : 06.1f}s',
        f'GRAD: {grad.item() : .2e}',
        f'RMSE: {np.sqrt(loss).item() : .2e}',
        f'L2-DR-TRN: {metrics_trn.direct.l2 * 100 : .2f}%',
        f'L1-DR: {metrics_val.direct.l1 * 100 : .2f}%',
        f'L2-DR: {metrics_val.direct.l2 * 100 : .2f}%',
        f'L1-FN: {metrics_val.final.l1 * 100 : .2f}%',
        f'L2-FN: {metrics_val.final.l2 * 100 : .2f}%',
      ]))

      with disable_logging(level=logging.FATAL):
        checkpoint_metrics = {
          'loss': loss.item(),
          'train': metrics_trn.to_dict(),
          'valid': metrics_val.to_dict(),
        }
        # Store the state and the metrics
        step = epochs_before + epoch
        checkpoint_manager.save(
          step=step,
          items={'state': jax.device_get(unreplicate(state)),},
          metrics=checkpoint_metrics,
          save_kwargs={'save_args': checkpointer_save_args}
        )
        with open(DIR / 'metrics' / f'{str(step)}.json', 'w') as f:
          json.dump(checkpoint_metrics, f)

    else:
      # Log the results
      time_tot = time() - time_int
      logging.info('\t'.join([
        f'DRCT: {direct_steps : 02d}',
        f'URLL: {unroll_steps : 02d}',
        f'EPCH: {epochs_before + epoch : 04d}/{FLAGS.epochs : 04d}',
        f'LR: {state.opt_state[-1].hyperparams["learning_rate"][0].item() : .2e}',
        f'TIME: {time_tot : 06.1f}s',
        f'GRAD: {grad.item() : .2e}',
        f'RMSE: {np.sqrt(loss).item() : .2e}',
      ]))


  return unreplicate(state)

def get_model(model_configs: Mapping[str, Any]) -> AbstractOperator:

  model = MPGNO(
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

  # Check the inputs
  if not FLAGS.datetime:
    FLAGS.datetime = datetime.now().strftime('%Y%m%d-%H%M%S')
  assert FLAGS.jump_steps <= MAX_JUMP_STEPS
  assert (IDX_FN % FLAGS.jump_steps) == 0

  # Initialize the random key
  key = jax.random.PRNGKey(FLAGS.seed)

  # Read the dataset
  subkey, key = jax.random.split(key)
  dataset = Dataset(
    key=subkey,
    datadir=FLAGS.datadir,
    datapath=FLAGS.datapath,
    n_train=FLAGS.n_train,
    n_valid=FLAGS.n_valid,
    downsample_factor=TIME_DOWNSAMPLE_FACTOR,
    cutoff=(IDX_FN + MAX_JUMP_STEPS),
    preload=True,
    include_passive_variables=False,
  )
  dataset.compute_stats(
    axes=(0, 1, 2, 3),
    derivs_degree=0,
    residual_steps=(FLAGS.direct_steps * FLAGS.jump_steps),
    skip_residual_steps=FLAGS.jump_steps,
  )

  # Read the checkpoint
  if FLAGS.params:
    DIR_OLD_EXPERIMENT = DIR_EXPERIMENTS / FLAGS.params
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    step = orbax.checkpoint.CheckpointManager(DIR_OLD_EXPERIMENT / 'checkpoints', orbax_checkpointer).latest_step()
    ckpt = orbax_checkpointer.restore(directory=(DIR_OLD_EXPERIMENT / 'checkpoints' / str(step) / 'default'))
    state = ckpt['state']
    params = state['params']
    with open(DIR_OLD_EXPERIMENT / 'configs.json', 'rb') as f:
      model_kwargs = json.load(f)['model_configs']
  else:
    params = None
    model_kwargs = None

  # Get the model
  if not model_kwargs:
    model_kwargs = dict(
      num_outputs=dataset.shape[-1],
      num_grid_nodes=dataset.shape[2:4],
      num_mesh_nodes=(FLAGS.num_mesh_nodes, FLAGS.num_mesh_nodes),
      use_tau=True,
      use_t=True,
      conditional_normalization=False,
      conditional_norm_latent_size=4,
      conditional_norm_unique=True,
      conditional_norm_nonlinear=True,
      deriv_degree=FLAGS.deriv_degree,
      latent_size=FLAGS.latent_size,
      num_mlp_hidden_layers=FLAGS.num_mlp_hidden_layers,
      num_message_passing_steps=FLAGS.num_message_passing_steps,
      num_message_passing_steps_grid=FLAGS.num_message_passing_steps_grid,
      overlap_factor_grid2mesh=FLAGS.overlap_factor_grid2mesh,
      overlap_factor_mesh2grid=FLAGS.overlap_factor_mesh2grid,
      num_multimesh_levels=FLAGS.num_multimesh_levels,
      node_coordinate_freqs=FLAGS.node_coordinate_freqs,
      p_dropout_edges_grid2mesh=FLAGS.p_dropout_edges_grid2mesh,
      p_dropout_edges_mesh2grid=FLAGS.p_dropout_edges_mesh2grid,
    )
  model = get_model(model_kwargs)

  # Store the configurations
  DIR = DIR_EXPERIMENTS / f'E{FLAGS.exp}' / FLAGS.datapath / FLAGS.datetime
  DIR.mkdir(parents=True)
  logging.info(f'Experiment stored in {DIR.relative_to(DIR_EXPERIMENTS).as_posix()}')
  flags = {f: FLAGS.get_flag_value(f, default=None) for f in FLAGS}
  with open(DIR / 'configs.json', 'w') as f:
    json.dump(fp=f,
      obj={'flags': flags, 'model_configs': model.configs},
      indent=2,
    )
  # Store the statistics
  with open(DIR / 'stats.pkl', 'wb') as f:
    pickle.dump(file=f, obj=dataset.stats)

  # Split the epochs
  epochs_u00 = int(FLAGS.epochs // (1 + .2 * FLAGS.unroll_steps))
  if FLAGS.unroll_steps:
    epochs_uxx = int((FLAGS.epochs - epochs_u00) // FLAGS.unroll_steps)
    epochs_uff = epochs_uxx + (FLAGS.epochs - epochs_u00) % FLAGS.unroll_steps
  # TRY: Allocate more epochs to the final direct_steps
  epochs_u00_dxx = epochs_u00 // FLAGS.direct_steps
  epochs_u00_dff = epochs_u00_dxx + epochs_u00 % FLAGS.direct_steps

  # Initialzize the model or use the loaded parameters
  if not params:
    _, sample_spec = dataset.sample
    num_grid_points = dataset.shape[2:4]
    num_vars = dataset.shape[-1]
    model_init_kwargs = dict(
      u_inp=jnp.ones(shape=(FLAGS.batch_size, 1, *num_grid_points, num_vars)),
      t_inp=jnp.zeros(shape=(FLAGS.batch_size, 1)),
      tau=jnp.ones(shape=(FLAGS.batch_size, 1)),
      specs=(jnp.ones_like(sample_spec).repeat(FLAGS.batch_size, axis=0)
        if (sample_spec is not None) else None),
    )
    subkey, key = jax.random.split(key)
    variables = jax.jit(model.init)(subkey, **model_init_kwargs)
    params = variables['params']

  # Calculate the total number of parameters
  n_model_parameters = np.sum(
  jax.tree_util.tree_flatten(jax.tree_map(lambda x: np.prod(x.shape).item(), params))[0]).item()
  logging.info(f'Total number of trainable paramters: {n_model_parameters}')

  # Train the model without unrolling
  schedule_direct_steps = False
  epochs_trained = 0
  num_batches = dataset.nums['train'] // FLAGS.batch_size
  num_times = dataset.shape[1] // FLAGS.jump_steps
  unroll_offset = FLAGS.unroll_steps * FLAGS.direct_steps
  num_lead_times = num_times - 1 - unroll_offset
  assert num_lead_times > 0
  num_lead_times_full = max(0, num_times - FLAGS.direct_steps - unroll_offset)
  num_lead_times_part = num_lead_times - num_lead_times_full
  transition_steps = 0
  for _d in (range(1, FLAGS.direct_steps+1) if schedule_direct_steps else [FLAGS.direct_steps]):
    num_valid_pairs_d = (
      num_lead_times_full * _d
      + (num_lead_times_part * (num_lead_times_part+1) // 2)
    )
    if schedule_direct_steps:
      epochs_d = (epochs_u00_dff if (_d == FLAGS.direct_steps) else epochs_u00_dxx)
    else:
      epochs_d = epochs_u00
    transition_steps +=  epochs_d * num_batches * FLAGS.jump_steps * num_valid_pairs_d

  pct_start = .02  # Warmup cosine onecycle
  pct_final = .1   # Final exponential decay
  lr = optax.join_schedules(
    schedules=[
      optax.cosine_onecycle_schedule(
        transition_steps=((1 - pct_final) * transition_steps),
        peak_value=FLAGS.lr_peak,
        pct_start=(pct_start / (1 - pct_final)),
        div_factor=(FLAGS.lr_peak / FLAGS.lr_init),
        final_div_factor=(FLAGS.lr_init / FLAGS.lr_base),
      ),
      optax.exponential_decay(
        transition_steps=(pct_final * transition_steps),
        init_value=FLAGS.lr_base,
        decay_rate=(FLAGS.lr_lowr / FLAGS.lr_base) if FLAGS.lr_lowr else 1,
      ),
    ],
    boundaries=[int((1 - pct_final) * transition_steps),],
  )
  tx = optax.chain(
    optax.inject_hyperparams(optax.adamw)(learning_rate=lr, weight_decay=1e-08),
  )
  state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
  for _d in (range(1, FLAGS.direct_steps+1) if schedule_direct_steps else [FLAGS.direct_steps]):
    key, subkey = jax.random.split(key)
    if schedule_direct_steps:
      epochs = (epochs_u00_dff if (_d == FLAGS.direct_steps) else epochs_u00_dxx)
    else:
      epochs = epochs_u00
    state = train(
      key=subkey,
      model=model,
      state=state,
      dataset=dataset,
      jump_steps=FLAGS.jump_steps,
      direct_steps=_d,
      unroll_steps=0,
      epochs=epochs,
      epochs_before=epochs_trained,
    )
    epochs_trained += epochs

  # Train the model with unrolling
  lr = FLAGS.lr_base
  tx = optax.chain(
    optax.inject_hyperparams(optax.adamw)(learning_rate=lr, weight_decay=1e-08),
  )
  state = TrainState.create(apply_fn=model.apply, params=state.params, tx=tx)
  for _u in range(1, FLAGS.unroll_steps+1):
    key, subkey = jax.random.split(key)
    epochs = (epochs_uff if (_u == FLAGS.unroll_steps) else epochs_uxx)
    state = train(
      key=subkey,
      model=model,
      state=state,
      dataset=dataset,
      jump_steps=FLAGS.jump_steps,
      direct_steps=FLAGS.direct_steps,
      unroll_steps=_u,
      epochs=epochs,
      epochs_before=epochs_trained,
    )
    epochs_trained += epochs

if __name__ == '__main__':
  logging.set_verbosity('info')
  app.run(main)
