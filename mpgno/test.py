import json
import pickle
import shutil
from typing import Tuple, Type, Mapping, Callable, Any, Sequence
from time import time

import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint
import pandas as pd
import scipy.ndimage
import matplotlib.pyplot as plt
from absl import app, flags, logging
from flax.training.common_utils import shard, shard_prng_key
from flax.jax_utils import replicate

from mpgno.dataset import Dataset
from mpgno.experiments import DIR_EXPERIMENTS
from mpgno.metrics import rel_lp_error_norm
from mpgno.models.mpgno import AbstractOperator, MPGNO
from mpgno.models.unet import UNet
from mpgno.plot import plot_estimations, plot_ensemble, plot_error_vs_time
from mpgno.stepping import Stepper, TimeDerivativeUpdater, ResidualUpdater, OutputUpdater
from mpgno.stepping import AutoregressiveStepper
from mpgno.utils import Array, disable_logging, profile


NUM_DEVICES = jax.local_device_count()
IDX_FN = 14

FLAGS = flags.FLAGS
flags.DEFINE_string(name='exp', default=None, required=True,
  help='Relative path of the experiment'
)
flags.DEFINE_string(name='datadir', default=None, required=True,
  help='Path of the folder containing the datasets'
)
flags.DEFINE_integer(name='batch_size', default=4, required=False,
  help='Size of a batch of training samples'
)
flags.DEFINE_integer(name='n_test', default=(2**8), required=False,
  help='Number of test samples'
)
flags.DEFINE_boolean(name='profile', default=False, required=False,
  help='If passed, inference is profiled with 1 GPU'
)
flags.DEFINE_boolean(name='resolution', default=False, required=False,
  help='If passed, estimations with different resolutions are computed'
)
flags.DEFINE_boolean(name='noise', default=False, required=False,
  help='If passed, estimations for noise control are computed'
)
flags.DEFINE_boolean(name='ensemble', default=False, required=False,
  help='If passed, ensemble samples are generated using model randomness'
)


def print_between_dashes(msg):
  logging.info('-' * 80)
  logging.info(msg)
  logging.info('-' * 80)

def change_resolution(u: Array, resolution: Tuple[int, int]) -> Array:
  """
  Changes the resolution of an input array on axes (2, 3).

  Args:
      u: The input array.
      resolution: The target resolution.

  Returns:
      The input array with the target resolution.
  """

  # Return if the resolution is fine
  if u.shape[2:4] == resolution:
    return u

  # Get the downsampling factor
  space_downsample_factor = [
    (dim_before / dim_after)
    for (dim_before, dim_after)
    in zip(u.shape[2:4], resolution)
  ]

  # Subsample if possible
  if all([(d % 1) == 0 for d in space_downsample_factor]):
    d = [int(d) for d in space_downsample_factor]
    return u[:, :, ::d[0], ::d[1]]
  # Interpolate otherwise
  else:
    d = [(1/d) for d in space_downsample_factor]
    return scipy.ndimage.zoom(u, zoom=(1, 1, d[0], d[1], 1), order=2)

def profile_inferrence(
  dataset: Dataset,
  model: AbstractOperator,
  stepping: Type[Stepper],
  state: dict,
  stats: dict,
  resolution: Tuple[int, int],
  p_edge_masking_grid2mesh: float,
  repeats: int = 10,
  jit: bool = True,
):

  # Configure and build new model
  model_configs = model.configs
  if isinstance(model, MPGNO):
    model_configs['num_grid_nodes'] = resolution
    model_configs['p_dropout_edges_grid2mesh'] = p_edge_masking_grid2mesh
    model_configs['p_dropout_edges_multimesh'] = 0.
    model_configs['p_dropout_edges_mesh2grid'] = 0.
  elif isinstance(model, UNet):
    pass
  else:
    raise NotImplementedError
  stepper = stepping(operator=model.__class__(**model_configs))

  apply_fn = stepper.apply
  if jit:
    apply_fn = jax.jit(apply_fn)

  # Get a batch and transform it
  batch_size_per_device = FLAGS.batch_size // NUM_DEVICES
  batch = next(dataset.batches(mode='test', batch_size=batch_size_per_device))
  u_inp = batch[:, [0]]
  u_inp = change_resolution(u_inp, resolution)

  # Profile compilation
  t_compilation = profile(
    f=apply_fn,
    kwargs=dict(
      variables={'params': state['params']},
      stats=stats,
      u_inp=u_inp,
      t_inp=0.,
      tau=1.,
    ),
    repeats=1,
  )

  # Profile inferrence after compilation
  t = profile(
    f=apply_fn,
    kwargs=dict(
      variables={'params': state['params']},
      stats=stats,
      u_inp=u_inp,
      t_inp=0.,
      tau=1.,
    ),
    repeats=repeats,
  ) / repeats

  general_info = [
    'NUMBER OF DEVICES: 1',
    f'BATCH SIZE PER DEVICE: {batch_size_per_device}',
    f'MODEL: {model.__class__.__name__}',
    f'RESOLUTION: {resolution}',
    f'p_edge_masking_grid2mesh: {p_edge_masking_grid2mesh}',
  ]

  times_info = [
    f'Compilation: {t_compilation : .2f}s',
    f'Inferrence: {t : .6f}s per batch',
    f'Inferrence: {t / batch_size_per_device : .6f}s per sample',
  ]

  # Print all messages in dashes
  def wrap_in_dashes(lines, width):
    return ['-' * width] + lines + ['-' * width]
  all_msgs = wrap_in_dashes(general_info, 80) + wrap_in_dashes(times_info, 80)
  for line in all_msgs:
    logging.info(line)

def get_direct_estimations(
  step: Stepper.apply,
  variables,
  stats,
  trajs: Array,
  tau: int,
  time_downsample_factor: int,
  key = None,
) -> Array:
  """Inputs are of shape [batch_size_per_device, ...]"""

  # Set lead times
  lead_times = jnp.arange(trajs.shape[1])
  batch_size = trajs.shape[0]

  # Get inputs for all lead times
  # -> [num_lead_times, batch_size_per_device, ...]
  u_inp = jax.vmap(
      lambda lt: jax.lax.dynamic_slice_in_dim(
        operand=trajs,
        start_index=(lt), slice_size=1, axis=1)
  )(lead_times)
  t_inp = lead_times.repeat(repeats=batch_size).reshape(-1, batch_size, 1)

  # Get model estimations
  def _use_step_on_mini_batches(carry, x):
    idx = carry
    _u_inp = u_inp[idx]
    _t_inp = t_inp[idx]
    _u_prd = step(
      variables=variables,
      stats=stats,
      u_inp=_u_inp,
      t_inp=(_t_inp / time_downsample_factor),
      tau=(tau / time_downsample_factor),
      key=key,
    )
    carry += 1
    return carry, _u_prd
  # -> [num_lead_times, batch_size_per_device, 1, ...]
  _, u_prd = jax.lax.scan(
    f=_use_step_on_mini_batches,
    init=0,
    xs=None,
    length=trajs.shape[1],
  )

  # Re-arrange
  # -> [batch_size_per_device, num_lead_times, ...]
  u_prd = u_prd.swapaxes(0, 1).squeeze(axis=2)

  return u_prd

def get_rollout_estimations(
  unroll: AutoregressiveStepper.unroll,
  num_steps: int,
  variables,
  stats,
  u_inp: Array,
  key = None,
):
  """Inputs are of shape [batch_size_per_device, ...]"""

  batch_size = u_inp.shape[0]
  rollout, _ = unroll(
    variables,
    stats,
    u_inp,
    jnp.array([0.]).repeat(repeats=(batch_size)).reshape(batch_size, 1),
    num_steps,
    key,
  )

  return rollout

def get_all_estimations(
  dataset: Dataset,
  model: AbstractOperator,
  stepping: Type[Stepper],
  state: dict,
  stats: dict,
  resolution_train: Tuple[int, int],
  train_flags: Mapping,
  taus_direct: Sequence[int] = [],
  taus_rollout: Sequence[int] = [],
  resolutions: Sequence[Tuple[int, int]] = [],
  noise_levels: Sequence[float] = [],
  p_edge_masking_grid2mesh: float = 0.,
):

  # Replicate state and stats
  # NOTE: Internally uses jax.device_put_replicate
  state = replicate(state)
  stats = replicate(stats)

  # Get pmapped version of the estimator functions
  _get_direct_estimations = jax.pmap(get_direct_estimations, static_broadcasted_argnums=(0,))
  _get_rollout_estimations = jax.pmap(get_rollout_estimations, static_broadcasted_argnums=(0, 1))

  def _get_estimations_in_batches(
    direct: bool,
    apply_fn: Callable,
    tau: int = None,
    transform: Callable[[Array], Array] = None,
  ):
    # Check inputs
    if direct:
      assert tau is not None

    # Loop over the batches
    u_prd = []
    for batch in dataset.batches(mode='test', batch_size=FLAGS.batch_size):
      # Transform the batch
      if transform is not None:
        batch = transform(batch)

      # Split the batch between devices
      # -> [NUM_DEVICES, batch_size_per_device, ...]
      batch = shard(batch)

      # Get the direct predictions
      if direct:
        u_prd_batch = _get_direct_estimations(
          apply_fn,
          variables={'params': state['params']},
          stats=stats,
          trajs=batch,
          tau=replicate(tau),
          time_downsample_factor=replicate(train_flags['time_downsample_factor']),
        )
        # Replace the tail of the predictions with the head of the input
        u_prd_batch = jnp.concatenate([batch[:, :, :tau], u_prd_batch[:, :, :-tau]], axis=2)

      # Get the rollout predictions
      else:
        num_times = batch.shape[2]
        u_prd_batch = _get_rollout_estimations(
          apply_fn,
          num_times,
          variables={'params': state['params']},
          stats=stats,
          u_inp=batch[:, :, [0]],
        )

      # Undo the split between devices
      u_prd_batch = u_prd_batch.reshape(FLAGS.batch_size, *u_prd_batch.shape[2:])

      # Append the prediction
      u_prd.append(u_prd_batch)

    # Concatenate the predictions
    u_prd = jnp.concatenate(u_prd, axis=0)

    return u_prd

  # Instantiate the steppers
  all_resolutions = set(resolutions + [resolution_train])
  steppers: dict[Any, Stepper] = {res: None for res in all_resolutions}
  apply_steppers_jit: dict[Any, Stepper.apply] = {res: None for res in all_resolutions}
  apply_steppers_twice_jit: dict[Any, Stepper.unroll] = {res: None for res in all_resolutions}
  unrollers: dict[Any, dict[Any, AutoregressiveStepper]] = {
    res: {tau: None for tau in taus_rollout} for res in all_resolutions}
  apply_unroll_jit: dict[Any, dict[Any, AutoregressiveStepper.unroll]] = {
    res: {tau: None for tau in taus_rollout} for res in all_resolutions}

  # Instantiate the steppers
  for resolution in all_resolutions:
    # Configure and build new model
    model_configs = model.configs
    if isinstance(model, MPGNO):
      model_configs['num_grid_nodes'] = resolution
      model_configs['p_dropout_edges_grid2mesh'] = p_edge_masking_grid2mesh
      model_configs['p_dropout_edges_multimesh'] = 0.
      model_configs['p_dropout_edges_mesh2grid'] = 0.
    elif isinstance(model, UNet):
      pass
    else:
      raise NotImplementedError

    steppers[resolution] = stepping(operator=model.__class__(**model_configs))
    apply_steppers_jit[resolution] = jax.jit(steppers[resolution].apply)
    def apply_steppers_twice(*args, **kwargs):
      return steppers[resolution].unroll(*args, **kwargs, num_steps=2)
    apply_steppers_twice_jit[resolution] = jax.jit(apply_steppers_twice)

    for tau_max in taus_rollout:
      unrollers[resolution][tau_max] = AutoregressiveStepper(
        stepper=steppers[resolution],
        tau_max=(tau_max / train_flags['time_downsample_factor']),
        tau_base=(1. / train_flags['time_downsample_factor'])
      )
      apply_unroll_jit[resolution][tau_max] = jax.jit(
        unrollers[resolution][tau_max].unroll, static_argnums=(4,))

  # Instantiate the outputs
  u_prd_tau = {'direct': {}, 'rollout': {}}
  u_prd_px = {'direct': {}, 'rollout': {}}
  u_prd_noise = {'direct': {}, 'rollout': {}}

  # Temporal continuity
  resolution = resolution_train
  for tau in taus_direct:
    if tau == .5:
      _apply_stepper = apply_steppers_twice_jit[resolution]
    else:
      _apply_stepper = apply_steppers_jit[resolution]
    t0 = time()
    u_prd_tau['direct'][tau] = {'resolution': resolution}
    u_prd_tau['direct'][tau]['u'] = _get_estimations_in_batches(
      direct=True,
      apply_fn=_apply_stepper,
      tau=(tau if tau != .5 else 1),
      transform=(lambda arr: change_resolution(arr, resolution)),
    )
    print_between_dashes(f'tau_direct={tau} \t TIME={time()-t0 : .4f}s')

  # Autoregressive rollout
  resolution = resolution_train
  for tau_max in taus_rollout:
    t0 = time()
    u_prd_tau['rollout'][tau_max] = {'resolution': resolution}
    u_prd_tau['rollout'][tau_max]['u'] = _get_estimations_in_batches(
      direct=False,
      apply_fn=apply_unroll_jit[resolution][tau_max],
      transform=(lambda arr: change_resolution(arr, resolution)),
    )
    print_between_dashes(f'tau_max={tau_max} \t TIME={time()-t0 : .4f}s')

  # Spatial continuity
  tau_max = train_flags['time_downsample_factor'] * train_flags['tau_max']
  tau = train_flags['time_downsample_factor']
  for resolution in resolutions:
    t0 = time()
    u_prd_px['direct'][resolution] = {'resolution': resolution}
    u_prd_px['direct'][resolution]['u'] = _get_estimations_in_batches(
      direct=True,
      apply_fn=apply_steppers_jit[resolution],
      tau=tau,
      transform=(lambda arr: change_resolution(arr, resolution)),
    )
    print_between_dashes(f'resolution={resolution} (direct) \t TIME={time()-t0 : .4f}s')
    t0 = time()
    u_prd_px['rollout'][resolution] = {'resolution': resolution}
    u_prd_px['rollout'][resolution]['u'] = _get_estimations_in_batches(
      direct=False,
      apply_fn=apply_unroll_jit[resolution][tau_max],
      transform=(lambda arr: change_resolution(arr, resolution)),
    )
    print_between_dashes(f'resolution={resolution} (rollout) \t TIME={time()-t0 : .4f}s')

  # Noise control
  tau_max = train_flags['time_downsample_factor'] * train_flags['tau_max']
  tau = train_flags['time_downsample_factor']
  resolution = resolution_train
  for noise_level in noise_levels:
    # Transformation
    def transform(arr):
      arr = change_resolution(arr, resolution)
      std_arr = np.std(arr, axis=(0, 2, 3), keepdims=True)
      arr += noise_level * np.random.normal(scale=std_arr, size=arr.shape)
      return arr
    # Direct estimations
    t0 = time()
    u_prd_noise['direct'][noise_level] = {'resolution': resolution}
    u_prd_noise['direct'][noise_level]['u'] = _get_estimations_in_batches(
      direct=True,
      apply_fn=apply_steppers_jit[resolution],
      tau=tau,
      transform=transform,
    )
    print_between_dashes(f'noise_level={noise_level} (direct) \t TIME={time()-t0 : .4f}s')
    # Rollout estimations
    t0 = time()
    u_prd_noise['rollout'][noise_level] = {'resolution': resolution}
    u_prd_noise['rollout'][noise_level]['u'] = _get_estimations_in_batches(
      direct=False,
      apply_fn=apply_unroll_jit[resolution][tau_max],
      transform=transform,
    )
    print_between_dashes(f'noise_level={noise_level} (rollout) \t TIME={time()-t0 : .4f}s')

  u_prd = {
    'tau': u_prd_tau,
    'px': u_prd_px,
    'noise': u_prd_noise,
  }

  return u_prd

def get_ensemble_estimations(
  repeats: int,
  dataset: Dataset,
  model: AbstractOperator,
  stepping: Type[Stepper],
  state: dict,
  stats: dict,
  resolution_train: Tuple[int, int],
  train_flags: Mapping,
  tau_max: int,
  p_edge_masking_grid2mesh: float,
  key,
):

  # Replicate state and stats
  # NOTE: Internally uses jax.device_put_replicate
  state = replicate(state)
  stats = replicate(stats)

  # Get pmapped version of the estimator functions
  _get_rollout_estimations = jax.pmap(get_rollout_estimations, static_broadcasted_argnums=(0, 1))

  def _get_estimations_in_batches(
    apply_fn: Callable,
    transform: Callable[[Array], Array] = None,
    key = None,
  ):
    # Loop over the batches
    u_prd = []
    for batch in dataset.batches(mode='test', batch_size=FLAGS.batch_size):
      # Transform the batch
      if transform is not None:
        batch = transform(batch)

      # Split the batch between devices
      # -> [NUM_DEVICES, batch_size_per_device, ...]
      batch = shard(batch)
      subkey, key = jax.random.split(key)
      subkey = shard_prng_key(subkey)

      # Get the rollout predictions
      num_times = batch.shape[2]
      u_prd_batch = _get_rollout_estimations(
        apply_fn,
        num_times,
        variables={'params': state['params']},
        stats=stats,
        u_inp=batch[:, :, [0]],
        key=subkey,
      )

      # Undo the split between devices
      u_prd_batch = u_prd_batch.reshape(FLAGS.batch_size, *u_prd_batch.shape[2:])

      # Append the prediction
      u_prd.append(u_prd_batch)

    # Concatenate the predictions
    u_prd = jnp.concatenate(u_prd, axis=0)

    return u_prd

  # Configure and build new model
  model_configs = model.configs
  if isinstance(model, MPGNO):
    model_configs['num_grid_nodes'] = resolution_train
    model_configs['p_dropout_edges_grid2mesh'] = p_edge_masking_grid2mesh
    model_configs['p_dropout_edges_multimesh'] = 0.
    model_configs['p_dropout_edges_mesh2grid'] = 0.
  elif isinstance(model, UNet):
    pass
  else:
    raise NotImplementedError

  stepper = stepping(operator=model.__class__(**model_configs))
  unroller = AutoregressiveStepper(
    stepper=stepper,
    tau_max=(tau_max / train_flags['time_downsample_factor']),
    tau_base=(1. / train_flags['time_downsample_factor'])
  )
  apply_unroll_jit = jax.jit(
    unroller.unroll, static_argnums=(4,))

  # Autoregressive rollout
  u_prd = []
  for i in range(repeats):
    t0 = time()
    subkey, key = jax.random.split(key)
    u_prd.append(
      _get_estimations_in_batches(
        apply_fn=apply_unroll_jit,
        transform=(lambda arr: change_resolution(arr, resolution_train)),
        key=subkey,
      )
    )
    print_between_dashes(f'ensemble_repeat={i} \t TIME={time()-t0 : .4f}s')
  u_prd = np.stack(u_prd)

  return u_prd

def main(argv):
  # Check the number of arguments
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Check the available devices
  # NOTE: We only support single-host training.
  with disable_logging():
    process_index = jax.process_index()
    process_count = jax.process_count()
    local_devices = jax.local_devices()
  logging.info('JAX host: %d / %d', process_index, process_count)
  logging.info('JAX local devices: %r', local_devices)
  assert process_count == 1

  # Read the arguments and check
  assert FLAGS.batch_size % NUM_DEVICES == 0
  datapath = '/'.join(FLAGS.exp.split('/')[1:-1])
  DIR = DIR_EXPERIMENTS / FLAGS.exp

  # Set the dataset
  dataset = Dataset(
    datadir=FLAGS.datadir,
    datapath=datapath,
    n_train=0,
    n_valid=0,
    n_test=FLAGS.n_test,
    preload=True,
    time_downsample_factor=1,
    space_downsample_factor=1,
  )
  dataset_small = Dataset(
    datadir=FLAGS.datadir,
    datapath=datapath,
    n_train=0,
    n_valid=0,
    n_test=min(4, FLAGS.n_test),
    preload=True,
    time_downsample_factor=1,
    space_downsample_factor=1,
  )

  # Read the stats
  with open(DIR / 'stats.pkl', 'rb') as f:
    stats = pickle.load(f)
  stats = {
      key: {k: jnp.array(v) for k, v in val.items()}
      for key, val in stats.items()
    }
  # Read the configs
  with open(DIR / 'configs.json', 'rb') as f:
    configs = json.load(f)
  time_downsample_factor = configs['flags']['time_downsample_factor']
  tau_max_train = configs['flags']['tau_max']
  model_name = configs['flags']['model'].upper()
  model_configs = configs['model_configs']
  resolution_train = tuple(configs['resolution'])
  # Read the state
  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  mngr = orbax.checkpoint.CheckpointManager(DIR / 'checkpoints')
  best_checkpointed_step = mngr.best_step()
  ckpt = orbax_checkpointer.restore(directory=(DIR / 'checkpoints' / str(best_checkpointed_step) / 'default'))
  state = jax.tree_map(jnp.array, ckpt['state'])

  # Set the stepper type
  if configs['flags']['stepper'] == 'out':
    stepping = OutputUpdater
  elif configs['flags']['stepper'] == 'res':
    stepping = ResidualUpdater
  elif configs['flags']['stepper'] == 'der':
    stepping = TimeDerivativeUpdater
  else:
    raise ValueError

  # Set the model
  if model_name == 'MPGNO':
    model_class = MPGNO
  elif model_name == 'UNET':
    model_class = UNet
  else:
    raise ValueError
  model = model_class(**model_configs)

  # Profile
  # NOTE: One compilation
  if FLAGS.profile:
    profile_inferrence(
      dataset=dataset,
      model=model,
      stepping=stepping,
      state=state,
      stats=stats,
      resolution=resolution_train,
      p_edge_masking_grid2mesh=0,
    )

  # Create a clean directory for tests
  DIR_TESTS = DIR / 'tests'
  DIR_FIGS = DIR_TESTS / 'figures'
  if DIR_TESTS.exists():
    shutil.rmtree(DIR_TESTS)
  DIR_TESTS.mkdir()
  DIR_FIGS.mkdir()

  # Set evaluation settings
  interpolate_tau = True
  tau_min = 1
  tau_max = time_downsample_factor * (tau_max_train + (tau_max_train > 1))
  taus_direct = [.5] + list(range(tau_min, tau_max + 1))
  if not interpolate_tau:
    taus_direct = [tau for tau in taus_direct if (tau % 2) == 0]
  # NOTE: One compilation per tau_rollout
  taus_rollout = [.5, 1] + [time_downsample_factor * d for d in range(1, tau_max_train+1)]
  # NOTE: Two compilations per resolution
  resolutions = [(px, px) for px in [32, 48, 64, 96, 128]] if FLAGS.resolution else []
  noise_levels = [0, .005, .01, .02] if FLAGS.noise else []

  # Set the groundtruth trajectories
  u_gtr = next(dataset.batches(mode='test', batch_size=dataset.nums['test']))
  u_gtr_small = next(dataset_small.batches(mode='test', batch_size=dataset_small.nums['test']))

  # Get model estimations with all settings
  u_prd = get_all_estimations(
    dataset=dataset,
    model=model,
    stepping=stepping,
    state=state,
    stats=stats,
    resolution_train=resolution_train,
    train_flags=configs['flags'],
    taus_direct=taus_direct,
    taus_rollout=taus_rollout,
    resolutions=resolutions,
    noise_levels=noise_levels,
    p_edge_masking_grid2mesh=0,
  )

  # Plot estimation visualizations
  (DIR_FIGS / 'samples').mkdir()
  for s in range(min(4, FLAGS.n_test)):
    fig, _ = plot_estimations(
      u_gtr=change_resolution(u_gtr, resolution_train),
      u_prd=u_prd['tau']['rollout'][time_downsample_factor]['u'],
      idx_out=IDX_FN,
      idx_inp=0,
      idx_traj=s,
      symmetric=dataset.metadata.signed,
      names=dataset.metadata.names,
    )
    fig.savefig(DIR_FIGS / 'samples' / f'rollout-fn-s{s:02d}.png')
    plt.close(fig)
    fig, _ = plot_estimations(
      u_gtr=change_resolution(u_gtr, resolution_train),
      u_prd=u_prd['tau']['rollout'][time_downsample_factor]['u'],
      idx_out=-1,
      idx_inp=0,
      idx_traj=s,
      symmetric=dataset.metadata.signed,
      names=dataset.metadata.names,
    )
    fig.savefig(DIR_FIGS / 'samples' / f'rollout-ex-s{s:02d}.png')
    plt.close(fig)

  # Compute the errors
  def _get_err_trajectory(_u_gtr, _u_prd, p):
    _err = [
      np.median(rel_lp_error_norm(_u_gtr[:, [idx_t]], _u_prd[:, [idx_t]], p=p)).item() * 100
      for idx_t in range(_u_gtr.shape[1])
    ]
    return _err
  errors_l1 = {
    key: {
      'direct': {
        str(subkey): _get_err_trajectory(
          _u_gtr=change_resolution(u_gtr, resolution=u_prd[key]['direct'][subkey]['resolution']),
          _u_prd=u_prd[key]['direct'][subkey]['u'],
          p=1,
        )
        for subkey in u_prd[key]['direct'].keys()
      },
      'rollout': {
        str(subkey): _get_err_trajectory(
          _u_gtr=change_resolution(u_gtr, resolution=u_prd[key]['rollout'][subkey]['resolution']),
          _u_prd=u_prd[key]['rollout'][subkey]['u'],
          p=1,
        )
        for subkey in u_prd[key]['rollout'].keys()
      },
    }
    for key in u_prd.keys()
  }
  errors_l2 = {
    key: {
      'direct': {
        str(subkey): _get_err_trajectory(
          _u_gtr=change_resolution(u_gtr, resolution=u_prd[key]['direct'][subkey]['resolution']),
          _u_prd=u_prd[key]['direct'][subkey]['u'],
          p=2,
        )
        for subkey in u_prd[key]['direct'].keys()
      },
      'rollout': {
        str(subkey): _get_err_trajectory(
          _u_gtr=change_resolution(u_gtr, resolution=u_prd[key]['rollout'][subkey]['resolution']),
          _u_prd=u_prd[key]['rollout'][subkey]['u'],
          p=2,
        )
        for subkey in u_prd[key]['rollout'].keys()
      },
    }
    for key in u_prd.keys()
  }

  # Store the errors
  with open(DIR_TESTS / 'errors.json', 'w') as f:
    json.dump(obj={'l1': errors_l1, 'l2': errors_l2}, fp=f)

  # Print minimum errors
  l1_final = min([errors_l1['tau']['rollout'][str(tau)][IDX_FN] for tau in taus_rollout])
  l2_final = min([errors_l2['tau']['rollout'][str(tau)][IDX_FN] for tau in taus_rollout])
  l1_extra = min([errors_l2['tau']['rollout'][str(tau)][-1] for tau in taus_rollout])
  l2_extra = min([errors_l2['tau']['rollout'][str(tau)][-1] for tau in taus_rollout])
  print_between_dashes(f'ERROR AT t={IDX_FN} \t l1: {l1_final : .2f}% \t l2: {l2_final : .2f}%')
  print_between_dashes(f'ERROR AT t={dataset.shape[1]-1} \t l1: {l1_extra : .2f}% \t l2: {l2_extra : .2f}%')

  # Plot the errors and store the plots
  (DIR_FIGS / 'errors').mkdir()
  def errors_to_df(_errors):
    df = pd.DataFrame(_errors)
    df['t'] = df.index
    df = df.melt(id_vars=['t'], value_name='error')
    return df
  # Set which errors to plot
  errors_plot = errors_l1
  # Temporal continuity
  g = plot_error_vs_time(
    df=errors_to_df(errors_plot['tau']['direct']),
    idx_fn=IDX_FN,
    variable_title='$\\tau$',
  )
  g.figure.savefig(DIR_FIGS / 'errors' / 'tau-direct.png')
  plt.close(g.figure)
  g = plot_error_vs_time(
    df=errors_to_df(errors_plot['tau']['rollout']),
    idx_fn=IDX_FN,
    variable_title='$\\tau_{max}$',
  )
  g.figure.savefig(DIR_FIGS / 'errors' / 'tau-rollout.png')
  plt.close(g.figure)
  # Noise control
  g = plot_error_vs_time(
    df=errors_to_df(errors_plot['noise']['direct']),
    idx_fn=IDX_FN,
    variable_title='$\\dfrac{\\sigma_{noise}}{\\sigma_{data}}$',
  )
  g.figure.savefig(DIR_FIGS / 'errors' / 'noise-direct.png')
  plt.close(g.figure)
  g = plot_error_vs_time(
    df=errors_to_df(errors_plot['noise']['rollout']),
    idx_fn=IDX_FN,
    variable_title='$\\dfrac{\\sigma_{noise}}{\\sigma_{data}}$',
  )
  g.figure.savefig(DIR_FIGS / 'errors' / 'noise-rollout.png')
  plt.close(g.figure)
  # Spatial continuity
  g = plot_error_vs_time(
    df=errors_to_df(errors_plot['px']['direct']),
    idx_fn=IDX_FN,
    variable_title='Resolution',
  )
  g.figure.savefig(DIR_FIGS / 'errors' / 'resolution-direct.png')
  plt.close(g.figure)
  g = plot_error_vs_time(
    df=errors_to_df(errors_plot['px']['rollout']),
    idx_fn=IDX_FN,
    variable_title='Resolution',
  )
  g.figure.savefig(DIR_FIGS / 'errors' / 'resolution-rollout.png')
  plt.close(g.figure)

  # Get ensemble estimations with the default settings
  # NOTE: One compilation
  if FLAGS.ensemble:
    key = jax.random.PRNGKey(45)
    subkey, key = jax.random.split(key)
    u_prd_ensemble = get_ensemble_estimations(
      repeats=20,
      dataset=dataset_small,
      model=model,
      stepping=stepping,
      state=state,
      stats=stats,
      resolution_train=resolution_train,
      train_flags=configs['flags'],
      tau_max=2,
      p_edge_masking_grid2mesh=0.5,
      key=subkey,
    )

    # Plot ensemble statistics
    (DIR_FIGS / 'ensemble').mkdir()
    for s in range(min(4, FLAGS.n_test)):
      fig, _ = plot_ensemble(
        u_gtr=change_resolution(u_gtr_small, resolution_train),
        u_ens=u_prd_ensemble,
        idx_out=IDX_FN,
        idx_traj=s,
        symmetric=dataset_small.metadata.signed,
        names=dataset_small.metadata.names,
      )
      fig.savefig(DIR_FIGS / 'ensemble' / f'rollout-fn-s{s:02d}.png')
      plt.close(fig)
      fig, _ = plot_ensemble(
        u_gtr=change_resolution(u_gtr_small, resolution_train),
        u_ens=u_prd_ensemble,
        idx_out=-1,
        idx_traj=s,
        symmetric=dataset_small.metadata.signed,
        names=dataset_small.metadata.names,
      )
      fig.savefig(DIR_FIGS / 'ensemble' / f'rollout-ex-s{s:02d}.png')
      plt.close(fig)

  print_between_dashes('DONE')

if __name__ == '__main__':
  logging.set_verbosity('info')
  app.run(main)
