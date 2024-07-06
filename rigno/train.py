import json
import pickle
import functools
from datetime import datetime
from time import time
from typing import Tuple, Any, Mapping, Iterable, Callable

import flax.linen as nn
import flax.typing
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
from absl import app, flags, logging
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from flax.training.common_utils import shard, shard_prng_key
from flax.jax_utils import replicate, unreplicate
from jax.tree_util import PyTreeDef

from rigno.dataset import Dataset
from rigno.experiments import DIR_EXPERIMENTS
from rigno.metrics import BatchMetrics, Metrics, EvalMetrics
from rigno.metrics import rel_lp_loss
from rigno.metrics import mse_error, rel_lp_error_per_var, rel_lp_error_norm
from rigno.models.rigno import RIGNO, AbstractOperator
from rigno.models.unet import UNet
from rigno.stepping import AutoregressiveStepper
from rigno.stepping import TimeDerivativeUpdater
from rigno.stepping import ResidualUpdater
from rigno.stepping import OutputUpdater
from rigno.test import get_direct_estimations
from rigno.utils import disable_logging, Array, shuffle_arrays, split_arrays, normalize


NUM_DEVICES = jax.local_device_count()
EVAL_FREQ = 50
IDX_FN = 14

FLAGS = flags.FLAGS

def define_flags():
  # FLAGS::general
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
  flags.DEFINE_integer(name='time_downsample_factor', default=2, required=False,
    help='Factor for time downsampling'
  )
  flags.DEFINE_integer(name='space_downsample_factor', default=1, required=False,
    help='Factor for space downsampling'
  )

  # FLAGS::training
  flags.DEFINE_string(name='model', default=None, required=True,
    help='Name of the model: ["RIGNO", "UNET"]'
  )
  flags.DEFINE_integer(name='batch_size', default=2, required=False,
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
  flags.DEFINE_string(name='stepper', default='der', required=False,
    help='Type of the stepper'
  )
  flags.DEFINE_integer(name='tau_max', default=1, required=False,
    help='Maximum number of time steps between input/output pairs during training'
  )
  flags.DEFINE_boolean(name='fractional', default=False, required=False,
    help='If passed, train with fractional time steps (unrolled)'
  )
  flags.DEFINE_integer(name='n_train', default=(2**9), required=False,
    help='Number of training samples'
  )
  flags.DEFINE_integer(name='n_valid', default=(2**8), required=False,
    help='Number of validation samples'
  )
  flags.DEFINE_integer(name='n_test', default=(2**8), required=False,
    help='Number of test samples'
  )

  # FLAGS::model::RIGNO
  flags.DEFINE_integer(name='num_mesh_nodes', default=64, required=False,
    help='Number of mesh nodes in each dimension'
  )
  flags.DEFINE_float(name='overlap_factor_grid2mesh', default=4.0, required=False,
    help='Overlap factor for grid2mesh edges (encoder)'
  )
  flags.DEFINE_float(name='overlap_factor_mesh2grid', default=4.0, required=False,
    help='Overlap factor for mesh2grid edges (decoder)'
  )
  flags.DEFINE_integer(name='num_multimesh_levels', default=4, required=False,
    help='Number of multimesh connection levels (processor)'
  )
  flags.DEFINE_integer(name='node_coordinate_freqs', default=2, required=False,
    help='Number of frequencies for encoding periodic node coordinates'
  )
  flags.DEFINE_integer(name='node_latent_size', default=128, required=False,
    help='Size of latent node features'
  )
  flags.DEFINE_integer(name='edge_latent_size', default=128, required=False,
    help='Size of latent edge features'
  )
  flags.DEFINE_integer(name='num_mlp_hidden_layers', default=1, required=False,
    help='Number of hidden layers of all MLPs'
  )
  flags.DEFINE_integer(name='mlp_hidden_size', default=128, required=False,
    help='Size of latent edge features'
  )
  flags.DEFINE_integer(name='num_message_passing_steps', default=18, required=False,
    help='Number of message-passing steps in the processor'
  )
  flags.DEFINE_integer(name='num_message_passing_steps_grid', default=0, required=False,
    help='Number of message-passing steps in the decoder'
  )
  flags.DEFINE_float(name='p_dropout_edges_grid2mesh', default=0.5, required=False,
    help='Probability of dropping out edges of grid2mesh'
  )
  flags.DEFINE_float(name='p_dropout_edges_multimesh', default=0.5, required=False,
    help='Probability of dropping out edges of the multi-mesh'
  )
  flags.DEFINE_float(name='p_dropout_edges_mesh2grid', default=0.5, required=False,
    help='Probability of dropping out edges of mesh2grid'
  )

  # FLAGS::model::UNET
  flags.DEFINE_integer(name='unet_features', default=1, required=False,
    help='Number of features (channels)'
  )

def train(
  key: flax.typing.PRNGKey,
  model: nn.Module,
  state: TrainState,
  dataset: Dataset,
  tau_max: int,
  unroll: bool,
  epochs: int,
  epochs_before: int = 0,
  loss_fn: Callable = rel_lp_loss,
) -> TrainState:
  """Trains a model and returns the state."""

  # Set constants
  num_samples_trn = dataset.nums['train']
  num_times = dataset.shape[1]
  num_grid_points = dataset.shape[2:4]
  num_vars = dataset.shape[-1]
  assert num_samples_trn % FLAGS.batch_size == 0
  num_batches = num_samples_trn // FLAGS.batch_size
  assert FLAGS.batch_size % NUM_DEVICES == 0
  batch_size_per_device = FLAGS.batch_size // NUM_DEVICES
  evaluation_frequency = (
    (FLAGS.epochs // EVAL_FREQ) if (FLAGS.epochs >= EVAL_FREQ)
    else 1
  )

  # Store the initial time
  time_int_pre = time()

  # Define the permissible lead times
  num_lead_times = num_times - 1
  assert num_lead_times > 0
  assert tau_max < num_times
  num_lead_times_full = max(0, num_times - tau_max)
  num_lead_times_part = num_lead_times - num_lead_times_full
  num_valid_pairs = (
    num_lead_times_full * tau_max
    + (num_lead_times_part * (num_lead_times_part+1) // 2)
  )
  lead_times = jnp.arange(num_times - 1)

  # Define the autoregressive predictor
  if FLAGS.stepper == 'der':
    stepper = TimeDerivativeUpdater(operator=model)
  elif FLAGS.stepper == 'res':
    stepper = ResidualUpdater(operator=model)
  elif FLAGS.stepper == 'out':
    stepper = OutputUpdater(operator=model)
  else:
    raise ValueError
  one_step_predictor = AutoregressiveStepper(
    stepper=stepper,
    tau_max=1,
    tau_base=1.,
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
  ) -> Tuple[TrainState, Array, Array]:
    """Loads a batch, normalizes it, updates the state based on it, and returns it."""

    def _update_state_per_subbatch(
      key: flax.typing.PRNGKey,
      state: TrainState,
      u_inp: Array,
      t_inp: Array,
      u_tgt: Array,
      tau: Array,
    ) -> Tuple[TrainState, Array, PyTreeDef]:
      # NOTE: INPUT SHAPES [batch_size_per_device, ...]

      def _get_loss_and_grads(
        key: flax.typing.PRNGKey,
        params: flax.typing.Collection,
        u_inp: Array,
        t_inp: Array,
        u_tgt: Array,
        tau: Array,
      ) -> Tuple[Array, PyTreeDef]:
        """
        Computes the loss and the gradients of the loss w.r.t the parameters.
        """

        def _compute_loss(
          params: flax.typing.Collection,
          u_inp: Array,
          t_inp: Array,
          tau: Array,
          u_tgt: Array,
          key: flax.typing.PRNGKey,
        ) -> Array:
          """Computes the prediction of the model and returns its loss."""

          variables = {'params': params}

          # Get the output
          key, subkey = jax.random.split(key)
          _loss_inputs = stepper.get_loss_inputs(
            variables=variables,
            stats=stats,
            u_inp=u_inp,
            t_inp=t_inp,
            u_tgt=u_tgt,
            tau=tau,
            key=subkey,
          )

          return loss_fn(*_loss_inputs)

        if unroll:
          # Split tau for unrolling
          key, subkey = jax.random.split(key)
          tau_cutoff = .2
          tau_mid = tau_cutoff + jax.random.uniform(key=subkey, shape=tau.shape) * (tau - 2 * tau_cutoff)
          # Get intermediary output
          key, subkey = jax.random.split(key)
          u_int = stepper.apply(
            variables={'params': params},
            stats=stats,
            u_inp=u_inp,
            t_inp=t_inp,
            tau=tau_mid,
            key=subkey,
          )
          t_int = t_inp + tau_mid
        else:
          tau_mid = 0.
          u_int = u_inp
          t_int = t_inp

        # Compute gradients
        key, subkey = jax.random.split(key)
        loss, grads = jax.value_and_grad(_compute_loss)(
          params, u_int, t_int, (tau - tau_mid), u_tgt, key=subkey)

        return loss, grads

      # Update state, loss, and gradients
      _loss, _grads = _get_loss_and_grads(
        key=key,
        params=state.params,
        u_inp=u_inp,
        t_inp=t_inp,
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
    u_inp_batch = jax.vmap(
        lambda lt: jax.lax.dynamic_slice_in_dim(
          operand=trajs,
          start_index=(lt), slice_size=1, axis=1)
    )(lead_times)
    t_inp_batch = jax.vmap(
        lambda lt: jax.lax.dynamic_slice_in_dim(
          operand=times,
          start_index=(lt), slice_size=1, axis=1)
    )(lead_times)
    u_tgt_batch = jax.vmap(
        lambda lt: jax.lax.dynamic_slice_in_dim(
          operand=jnp.concatenate([trajs, jnp.zeros_like(trajs)], axis=1),
          start_index=(lt+1), slice_size=tau_max, axis=1)
    )(lead_times)

    # Repeat inputs along the time axis to match with u_tgt
    # -> [num_lead_times, batch_size_per_device, tau_max, ...]
    u_inp_batch = jnp.tile(u_inp_batch, reps=(1, 1, tau_max, 1, 1, 1))
    t_inp_batch = jnp.tile(t_inp_batch, reps=(1, 1, tau_max, 1, 1, 1))
    tau_batch = jnp.tile(
      (jnp.arange(1, tau_max+1)).reshape(1, 1, tau_max, 1),
      reps=(num_lead_times, batch_size_per_device, 1, 1)
    )

    # Put all pairs along the batch axis
    # -> [batch_size_per_device * num_lead_times * tau_max, ...]
    u_inp_batch = u_inp_batch.reshape((num_lead_times*batch_size_per_device*tau_max), 1, *num_grid_points, num_vars)
    t_inp_batch = t_inp_batch.reshape((num_lead_times*batch_size_per_device*tau_max), 1)
    tau_batch = tau_batch.reshape((num_lead_times*batch_size_per_device*tau_max), 1)
    u_tgt_batch = u_tgt_batch.reshape((num_lead_times*batch_size_per_device*tau_max), 1, *num_grid_points, num_vars)

    # Remove the invalid pairs
    # -> [batch_size_per_device * num_valid_pairs, ...]
    offset_full_lead_times = (num_times - tau_max) * tau_max * batch_size_per_device
    idx_invalid_pairs = np.array([
      (offset_full_lead_times + (_d * batch_size_per_device + _b) * tau_max - (_n + 1))
      for _d in range(tau_max - 1)
      for _b in range(1, batch_size_per_device + 1)
      for _n in range(_d + 1)
    ]).astype(int)
    u_inp_batch = jnp.delete(u_inp_batch, idx_invalid_pairs, axis=0)
    t_inp_batch = jnp.delete(t_inp_batch, idx_invalid_pairs, axis=0)
    tau_batch = jnp.delete(tau_batch, idx_invalid_pairs, axis=0)
    u_tgt_batch = jnp.delete(u_tgt_batch, idx_invalid_pairs, axis=0)

    # Shuffle and split the pairs
    # -> [num_valid_pairs, batch_size_per_device, ...]
    num_valid_pairs = u_tgt_batch.shape[0] // batch_size_per_device
    key, subkey = jax.random.split(key)
    u_inp_batch, t_inp_batch, tau_batch, u_tgt_batch = shuffle_arrays(
      subkey, [u_inp_batch, t_inp_batch, tau_batch, u_tgt_batch])
    u_inp_batch, t_inp_batch, tau_batch, u_tgt_batch = split_arrays(
      [u_inp_batch, t_inp_batch, tau_batch, u_tgt_batch], size=batch_size_per_device)

    # Add loss and gradients for each subbatch
    def _update_state(i, carry):
      # Update state, loss, and gradients
      _state, _loss_carried, _grads_carried, _key_carried = carry
      _key_updated, _subkey = jax.random.split(_key_carried)
      _state, _loss_subbatch, _grads_subbatch = _update_state_per_subbatch(
        key=_subkey,
        state=_state,
        u_inp=u_inp_batch[i],
        t_inp=t_inp_batch[i],
        u_tgt=u_tgt_batch[i],
        tau=tau_batch[i],
      )
      # Update the carried loss and gradients of the subbatch
      _loss_updated = _loss_carried + _loss_subbatch / num_valid_pairs
      _grads_updated = jax.tree_util.tree_map(
        lambda g_old, g_new: (g_old + g_new / num_valid_pairs),
        _grads_carried,
        _grads_subbatch,
      )

      return _state, _loss_updated, _grads_updated, _key_updated

    # Loop over the pairs
    _init_state = state
    _init_loss = 0.
    _init_grads = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), state.params)
    key, _init_key = jax.random.split(key)
    state, loss, grads, _ = jax.lax.fori_loop(
      lower=0,
      upper=num_valid_pairs,
      body_fun=_update_state,
      init_val=(_init_state, _init_loss, _init_grads, _init_key)
    )

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
      # -> [batch_size, num_times, ...]
      trajs = batch
      times = jnp.tile(jnp.arange(trajs.shape[1]), reps=(trajs.shape[0], 1))

      # Split the batch between devices
      # -> [NUM_DEVICES, batch_size_per_device, ...]
      trajs = shard(trajs)
      times = shard(times)

      # Get loss and updated state
      subkey, key = jax.random.split(key)
      subkey = shard_prng_key(subkey)
      state, loss, grads = _train_one_batch(subkey, state, stats, trajs, times)
      # NOTE: Using the first element of replicated loss and grads
      loss_epoch += loss[0] * FLAGS.batch_size / num_samples_trn
      grad_epoch += np.mean(jax.tree_util.tree_flatten(
        jax.tree_util.tree_map(jnp.mean, jax.tree_util.tree_map(lambda g: jnp.abs(g[0]), grads)))[0]) / num_batches

    return state, loss_epoch, grad_epoch

  @functools.partial(jax.pmap, static_broadcasted_argnums=(0,))
  def _evaluate_direct_prediction(
    tau: int,
    state: TrainState,
    stats,
    trajs: Array,
  ) -> Mapping:

    if tau < 1:
      step = lambda *args, **kwargs: stepper.unroll(*args, **kwargs, num_steps=int(1 / tau))
      _tau = 1
    else:
      step = stepper.apply
      _tau = tau

    u_prd = get_direct_estimations(
      step=step,
      variables={'params': state.params},
      stats=stats,
      trajs=trajs,
      tau=_tau,
      time_downsample_factor=1,
    )

    # Get mean errors per each sample in the batch
    batch_metrics = BatchMetrics(
      mse=mse_error(trajs[:, _tau:], u_prd[:, :-_tau]),
      l1=rel_lp_error_norm(trajs[:, _tau:], u_prd[:, :-_tau], p=1),
      l2=rel_lp_error_norm(trajs[:, _tau:], u_prd[:, :-_tau], p=2),
    )

    return batch_metrics.__dict__

  @jax.pmap
  def _evaluate_rollout_prediction(
    state: TrainState,
    stats,
    trajs: Array,
    times: Array,
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
    _predictor = one_step_predictor
    u_prd, _ = _predictor.unroll(
      variables=variables,
      stats=stats,
      u_inp=u_inp,
      t_inp=t_inp,
      num_steps=num_times,
    )

    # Calculate the errors
    batch_metrics = BatchMetrics(
      mse=mse_error(u_tgt, u_prd),
      l1=rel_lp_error_norm(u_tgt, u_prd, p=1),
      l2=rel_lp_error_norm(u_tgt, u_prd, p=2),
    )


    return batch_metrics.__dict__

  @jax.pmap
  def _evaluate_final_prediction(
    state: TrainState,
    stats,
    trajs: Array,
    times: Array,
  ) -> Mapping:

    # Set input and target
    idx_fn = IDX_FN // FLAGS.time_downsample_factor
    u_inp = trajs[:, :1]
    t_inp = times[:, :1]
    u_tgt = trajs[:, (idx_fn):(idx_fn+1)]

    # Get prediction at the final step
    _predictor = one_step_predictor
    _num_jumps = idx_fn // _predictor.num_steps_direct
    _num_direct_steps = idx_fn % _predictor.num_steps_direct
    variables = {'params': state.params}
    u_prd = _predictor.jump(
      variables=variables,
      stats=stats,
      u_inp=u_inp,
      t_inp=t_inp,
      num_jumps=_num_jumps,
    )
    if _num_direct_steps:
      _, u_prd = _predictor.unroll(
        variables=variables,
        stats=stats,
        u_inp=u_prd,
        t_inp=t_inp,
        num_steps=_num_direct_steps,
      )

    # Calculate the errors
    batch_metrics = BatchMetrics(
      mse=mse_error(u_tgt, u_prd),
      l1=rel_lp_error_norm(u_tgt, u_prd, p=1),
      l2=rel_lp_error_norm(u_tgt, u_prd, p=2),
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

    metrics_direct_tau_frac: list[BatchMetrics] = []
    metrics_direct_tau_min: list[BatchMetrics] = []
    metrics_direct_tau_max: list[BatchMetrics] = []
    metrics_rollout: list[BatchMetrics] = []
    metrics_final: list[BatchMetrics] = []

    for batch in batches:
      # Unwrap the batch
      trajs = batch
      times = jnp.tile(jnp.arange(trajs.shape[1]), reps=(trajs.shape[0], 1))

      # Split the batch between devices
      # -> [NUM_DEVICES, batch_size_per_device, ...]
      trajs = shard(trajs)
      times = shard(times)

      # Evaluate direct prediction
      if direct:
        # tau=.5
        batch_metrics_direct_tau_frac = _evaluate_direct_prediction(
          .5, state, stats, trajs,
        )
        batch_metrics_direct_tau_frac = BatchMetrics(**batch_metrics_direct_tau_frac)
        # Re-arrange the sub-batches gotten from each device
        batch_metrics_direct_tau_frac.reshape(shape=(batch_size_per_device * NUM_DEVICES, 1))
        # Append the metrics to the list
        metrics_direct_tau_frac.append(batch_metrics_direct_tau_frac)

        # tau=1
        batch_metrics_direct_tau_min = _evaluate_direct_prediction(
          1, state, stats, trajs,
        )
        batch_metrics_direct_tau_min = BatchMetrics(**batch_metrics_direct_tau_min)
        # Re-arrange the sub-batches gotten from each device
        batch_metrics_direct_tau_min.reshape(shape=(batch_size_per_device * NUM_DEVICES, 1))
        # Append the metrics to the list
        metrics_direct_tau_min.append(batch_metrics_direct_tau_min)

        # tau=tau_max
        batch_metrics_direct_tau_max = _evaluate_direct_prediction(
          FLAGS.tau_max, state, stats, trajs,
        )
        batch_metrics_direct_tau_max = BatchMetrics(**batch_metrics_direct_tau_max)
        # Re-arrange the sub-batches gotten from each device
        batch_metrics_direct_tau_max.reshape(shape=(batch_size_per_device * NUM_DEVICES, 1))
        # Append the metrics to the list
        metrics_direct_tau_max.append(batch_metrics_direct_tau_max)

      # Evaluate rollout prediction
      if rollout:
        batch_metrics_rollout = _evaluate_rollout_prediction(
          state, stats, trajs, times,
        )
        batch_metrics_rollout = BatchMetrics(**batch_metrics_rollout)
        # Re-arrange the sub-batches gotten from each device
        batch_metrics_rollout.reshape(shape=(batch_size_per_device * NUM_DEVICES, 1))
        # Compute and store metrics
        metrics_rollout.append(batch_metrics_rollout)

      # Evaluate final prediction
      if final:
        assert (IDX_FN // FLAGS.time_downsample_factor) < trajs.shape[2]
        batch_metrics_final = _evaluate_final_prediction(
          state, stats, trajs, times,
        )
        batch_metrics_final = BatchMetrics(**batch_metrics_final)
        # Re-arrange the sub-batches gotten from each device
        batch_metrics_final.reshape(shape=(batch_size_per_device * NUM_DEVICES, 1))
        # Append the errors to the list
        metrics_final.append(batch_metrics_final)

    # Aggregate over the batch dimension and compute norm per variable
    metrics_direct_tau_frac = Metrics(
      mse=jnp.median(jnp.concatenate([m.mse for m in metrics_direct_tau_frac]), axis=0).item(),
      l1=jnp.median(jnp.concatenate([m.l1 for m in metrics_direct_tau_frac]), axis=0).item(),
      l2=jnp.median(jnp.concatenate([m.l2 for m in metrics_direct_tau_frac]), axis=0).item(),
    ) if direct else None
    metrics_direct_tau_min = Metrics(
      mse=jnp.median(jnp.concatenate([m.mse for m in metrics_direct_tau_min]), axis=0).item(),
      l1=jnp.median(jnp.concatenate([m.l1 for m in metrics_direct_tau_min]), axis=0).item(),
      l2=jnp.median(jnp.concatenate([m.l2 for m in metrics_direct_tau_min]), axis=0).item(),
    ) if direct else None
    metrics_direct_tau_max = Metrics(
      mse=jnp.median(jnp.concatenate([m.mse for m in metrics_direct_tau_max]), axis=0).item(),
      l1=jnp.median(jnp.concatenate([m.l1 for m in metrics_direct_tau_max]), axis=0).item(),
      l2=jnp.median(jnp.concatenate([m.l2 for m in metrics_direct_tau_max]), axis=0).item(),
    ) if direct else None
    metrics_rollout = Metrics(
      mse=jnp.median(jnp.concatenate([m.mse for m in metrics_rollout]), axis=0).item(),
      l1=jnp.median(jnp.concatenate([m.l1 for m in metrics_rollout]), axis=0).item(),
      l2=jnp.median(jnp.concatenate([m.l2 for m in metrics_rollout]), axis=0).item(),
    ) if rollout else None
    metrics_final = Metrics(
      mse=jnp.median(jnp.concatenate([m.mse for m in metrics_final]), axis=0).item(),
      l1=jnp.median(jnp.concatenate([m.l1 for m in metrics_final]), axis=0).item(),
      l2=jnp.median(jnp.concatenate([m.l2 for m in metrics_final]), axis=0).item(),
    ) if final else None

    # Build the metrics object
    metrics = EvalMetrics(
      direct_tau_frac=(metrics_direct_tau_frac if direct else Metrics()),
      direct_tau_min=(metrics_direct_tau_min if direct else Metrics()),
      direct_tau_max=(metrics_direct_tau_max if direct else Metrics()),
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
    f'DRCT: {tau_max : 02d}',
    f'EPCH: {epochs_before : 04d}/{FLAGS.epochs : 04d}',
    f'LR: {state.opt_state[-1].hyperparams["learning_rate"][0].item() : .2e}',
    f'TIME: {time_tot_pre : 06.1f}s',
    f'GRAD: {0. : .2e}',
    f'LOSS: {0. : .2e}',
    f'DR-0.5: {metrics_val.direct_tau_frac.l1 * 100 : .2f}%',
    f'DR-1: {metrics_val.direct_tau_min.l1 * 100 : .2f}%',
    f'DR-{FLAGS.tau_max}: {metrics_val.direct_tau_max.l1 * 100 : .2f}%',
    f'FN: {metrics_val.final.l1 * 100 : .2f}%',
    f'TRN-DR-1: {metrics_trn.direct_tau_min.l1 * 100 : .2f}%',
    f'TRN-FN: {metrics_trn.final.l1 * 100 : .2f}%',
  ]))

  # Set up the checkpoint manager
  DIR = DIR_EXPERIMENTS / f'E{FLAGS.exp}' / FLAGS.datapath / FLAGS.datetime
  with disable_logging(level=logging.FATAL):
    (DIR / 'metrics').mkdir(exist_ok=True)
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpointer_options = orbax.checkpoint.CheckpointManagerOptions(
      max_to_keep=1,
      keep_period=None,
      best_fn=(lambda metrics: metrics['valid']['final']['l2']),
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
        f'DRCT: {tau_max : 02d}',
        f'EPCH: {epochs_before + epoch : 04d}/{FLAGS.epochs : 04d}',
        f'LR: {state.opt_state[-1].hyperparams["learning_rate"][0].item() : .2e}',
        f'TIME: {time_tot : 06.1f}s',
        f'GRAD: {grad.item() : .2e}',
        f'LOSS: {loss.item() : .2e}',
        f'DR-0.5: {metrics_val.direct_tau_frac.l1 * 100 : .2f}%',
        f'DR-1: {metrics_val.direct_tau_min.l1 * 100 : .2f}%',
        f'DR-{FLAGS.tau_max}: {metrics_val.direct_tau_max.l1 * 100 : .2f}%',
        f'FN: {metrics_val.final.l1 * 100 : .2f}%',
        f'TRN-DR-1: {metrics_trn.direct_tau_min.l1 * 100 : .2f}%',
        f'TRN-FN: {metrics_trn.final.l1 * 100 : .2f}%',
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
        f'DRCT: {tau_max : 02d}',
        f'EPCH: {epochs_before + epoch : 04d}/{FLAGS.epochs : 04d}',
        f'LR: {state.opt_state[-1].hyperparams["learning_rate"][0].item() : .2e}',
        f'TIME: {time_tot : 06.1f}s',
        f'GRAD: {grad.item() : .2e}',
        f'LOSS: {loss.item() : .2e}',
      ]))


  return unreplicate(state)

def get_model(model_name: str, model_configs: Mapping[str, Any], dataset: Dataset) -> AbstractOperator:
  """
  Build the model based on the given configurations.
  """

  # Check the inputs
  model_name = model_name.upper()
  assert model_name in ['RIGNO', 'UNET']

  # Set model kwargs
  if not model_configs:
    if model_name == 'RIGNO':
      model_configs = dict(
        num_outputs=dataset.shape[-1],
        num_grid_nodes=dataset.shape[2:4],
        num_mesh_nodes=(FLAGS.num_mesh_nodes, FLAGS.num_mesh_nodes),
        periodic=dataset.metadata.periodic,
        concatenate_tau=True,
        concatenate_t=True,
        conditional_normalization=True,
        conditional_norm_latent_size=16,
        node_latent_size=FLAGS.node_latent_size,
        edge_latent_size=FLAGS.edge_latent_size,
        num_mlp_hidden_layers=FLAGS.num_mlp_hidden_layers,
        mlp_hidden_size=FLAGS.mlp_hidden_size,
        num_message_passing_steps=FLAGS.num_message_passing_steps,
        num_message_passing_steps_grid=FLAGS.num_message_passing_steps_grid,
        overlap_factor_grid2mesh=FLAGS.overlap_factor_grid2mesh,
        overlap_factor_mesh2grid=FLAGS.overlap_factor_mesh2grid,
        num_multimesh_levels=FLAGS.num_multimesh_levels,
        node_coordinate_freqs=FLAGS.node_coordinate_freqs,
        p_dropout_edges_grid2mesh=FLAGS.p_dropout_edges_grid2mesh,
        p_dropout_edges_multimesh=FLAGS.p_dropout_edges_multimesh,
        p_dropout_edges_mesh2grid=FLAGS.p_dropout_edges_mesh2grid,
      )
    elif model_name == 'UNET':
      model_configs = dict(
        features=FLAGS.unet_features,
        outputs=dataset.shape[-1],
        conditional_norm_latent_size=16,
      )

  # Set the model class
  if model_name == 'RIGNO':
    model_class = RIGNO
  elif model_name == 'UNET':
    model_class = UNet

  model = model_class(**model_configs)

  return model

def main(argv):
  # Check the number of arguments
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
    n_test=FLAGS.n_test,
    time_downsample_factor=FLAGS.time_downsample_factor,
    space_downsample_factor=FLAGS.space_downsample_factor,
    cutoff=((IDX_FN // FLAGS.time_downsample_factor) + 1),
    preload=True,
    include_passive_variables=False,
  )
  dataset.compute_stats(
    axes=(0, 1, 2, 3),
    residual_steps=FLAGS.tau_max,
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
      old_configs = json.load(f)
      model_name = old_configs['flags']['model']
      model_configs = old_configs['model_configs']
  else:
    params = None
    model_name = FLAGS.model or 'RIGNO'
    model_configs = None

  # Get the model
  model = get_model(model_name, model_configs, dataset)

  # Store the configurations
  DIR = DIR_EXPERIMENTS / f'E{FLAGS.exp}' / FLAGS.datapath / FLAGS.datetime
  DIR.mkdir(parents=True)
  logging.info(f'Experiment stored in {DIR.relative_to(DIR_EXPERIMENTS).as_posix()}')
  flags = {f: FLAGS.get_flag_value(f, default=None) for f in FLAGS}
  with open(DIR / 'configs.json', 'w') as f:
    json.dump(fp=f,
      obj={'flags': flags, 'model_configs': model.configs, 'resolution': dataset.shape[2:4]},
      indent=2,
    )
  # Store the statistics
  with open(DIR / 'stats.pkl', 'wb') as f:
    pickle.dump(file=f, obj=dataset.stats)

  schedule_tau_max = True
  if (FLAGS.tau_max == 1):
    schedule_tau_max = False

  # Split the epochs
  if schedule_tau_max:
    epochs_warmup = int(.2 * FLAGS.epochs)
    epochs_dxx = epochs_warmup // (FLAGS.tau_max - 1)
    epochs_dff = (FLAGS.epochs - epochs_warmup) + epochs_warmup % (FLAGS.tau_max - 1)

  # Initialzize the model or use the loaded parameters
  if not params:
    num_grid_points = dataset.shape[2:4]
    num_vars = dataset.shape[-1]
    model_init_kwargs = dict(
      u_inp=jnp.ones(shape=(FLAGS.batch_size, 1, *num_grid_points, num_vars)),
      t_inp=jnp.zeros(shape=(FLAGS.batch_size, 1)),
      tau=jnp.ones(shape=(FLAGS.batch_size, 1)),
    )
    subkey, key = jax.random.split(key)
    variables = jax.jit(model.init)(subkey, **model_init_kwargs)
    params = variables['params']

  # Calculate the total number of parameters
  n_model_parameters = np.sum(
    jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.prod(x.shape).item(), params))[0]
  ).item()
  logging.info(f'Training a {model.__class__.__name__} with {n_model_parameters} parameters')

  # Train the model
  epochs_trained = 0
  num_batches = dataset.nums['train'] // FLAGS.batch_size
  num_times = dataset.shape[1]
  num_lead_times = num_times - 1
  assert num_lead_times > 0
  num_lead_times_full = max(0, num_times - FLAGS.tau_max)
  num_lead_times_part = num_lead_times - num_lead_times_full
  transition_steps = 0
  for _d in (range(1, FLAGS.tau_max+1) if schedule_tau_max else [FLAGS.tau_max]):
    num_valid_pairs_d = (
      num_lead_times_full * _d
      + (num_lead_times_part * (num_lead_times_part+1) // 2)
    )
    if schedule_tau_max:
      epochs_d = (epochs_dff if (_d == FLAGS.tau_max) else epochs_dxx)
    else:
      epochs_d = FLAGS.epochs
    transition_steps +=  epochs_d * num_batches * num_valid_pairs_d

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

  # Warm-up epochs
  if schedule_tau_max:
    for _d in range(1, FLAGS.tau_max):
      key, subkey = jax.random.split(key)
      state = train(
        key=subkey,
        model=model,
        state=state,
        dataset=dataset,
        tau_max=_d,
        unroll=False,
        epochs=epochs_dxx,
        epochs_before=epochs_trained,
      )
      epochs_trained += epochs_dxx

  # Split with and without unrolling
  if schedule_tau_max:
    epochs_rest = epochs_dff
  else:
    epochs_rest = FLAGS.epochs
  epochs_with_unrolling = int(.5 * epochs_rest)
  epochs_without_unrolling = epochs_rest - epochs_with_unrolling

  if FLAGS.fractional:
    # Train without unrolling
    state = train(
      key=subkey,
      model=model,
      state=state,
      dataset=dataset,
      tau_max=FLAGS.tau_max,
      unroll=False,
      epochs=epochs_without_unrolling,
      epochs_before=epochs_trained,
    )
    epochs_trained += epochs_without_unrolling
    # Train with unrolling
    logging.info('-' * 80)
    logging.info('WITH UNROLLING')
    logging.info('-' * 80)
    state = train(
      key=subkey,
      model=model,
      state=state,
      dataset=dataset,
      tau_max=FLAGS.tau_max,
      unroll=True,
      epochs=epochs_with_unrolling,
      epochs_before=epochs_trained,
    )
    epochs_trained += epochs_with_unrolling

  else:
    # Train without unrolling
    state = train(
      key=subkey,
      model=model,
      state=state,
      dataset=dataset,
      tau_max=FLAGS.tau_max,
      unroll=False,
      epochs=epochs_rest,
      epochs_before=epochs_trained,
    )
    epochs_trained += epochs_rest

if __name__ == '__main__':
  logging.set_verbosity('info')
  define_flags()
  app.run(main)
