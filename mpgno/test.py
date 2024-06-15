import functools
import json
import pickle
import shutil
from typing import Tuple, Type, Mapping, Callable, Any, Sequence

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

  # Instantiate the steppers
  all_resolutions = set(resolutions + [resolution_train])
  steppers: dict[Any, Stepper] = {res: None for res in all_resolutions}
  unrollers: dict[Any, dict[Any, AutoregressiveStepper]] = {
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

    for tau_max in taus_rollout:
      unrollers[resolution][tau_max] = AutoregressiveStepper(
        stepper=steppers[resolution],
        tau_max=(tau_max / train_flags['time_downsample_factor']),
        tau_base=(1. / train_flags['time_downsample_factor'])
      )

  @functools.partial(jax.pmap, static_broadcasted_argnums=(0,))
  def _get_direct_estimations(
    resolution: Tuple[int, int],
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
    # -> [num_lead_times, batch_size_per_device, 1, ...]
    def _use_step_on_mini_batches(_u_inp, _t_inp):
      return steppers[resolution].apply(
        variables=variables,
        stats=stats,
        u_inp=_u_inp,
        t_inp=(_t_inp / time_downsample_factor),
        tau=(tau / time_downsample_factor),
        key=key,
      )
    u_prd = jax.vmap(_use_step_on_mini_batches)(u_inp, t_inp)

    # Re-arrange
    # -> [batch_size_per_device, num_lead_times, ...]
    u_prd = u_prd.swapaxes(0, 1).squeeze(axis=2)

    return u_prd

  @functools.partial(jax.pmap, static_broadcasted_argnums=(0, 1, 2,))
  def _get_rollout_estimations(
    resolution: Tuple[int, int],
    tau_max: int,
    num_steps: int,
    variables,
    stats,
    u_inp: Array,
    key = None,
  ):
    """Inputs are of shape [batch_size_per_device, ...]"""

    batch_size = u_inp.shape[0]
    rollout, _ = unrollers[resolution][tau_max].unroll(
      variables,
      stats,
      u_inp,
      jnp.array([0.]).repeat(repeats=(batch_size)).reshape(batch_size, 1),
      num_steps,
      key,
    )

    return rollout

  def _get_estimations_in_batches(
    direct: bool,
    resolution: Tuple[int, int],
    tau: int,
    transform: Callable[[Array], Array] = None,
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

      # Get the direct predictions
      if direct:
        u_prd_batch = _get_direct_estimations(
          resolution,
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
        tau_max = tau
        u_prd_batch = _get_rollout_estimations(
          resolution,
          tau_max,
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

  # Instantiate the outputs
  u_prd_tau = {'direct': {}, 'rollout': {}}
  u_prd_px = {'direct': {}, 'rollout': {}}
  u_prd_noise = {'direct': {}, 'rollout': {}}

  # Temporal continuity
  resolution = resolution_train
  for tau in taus_direct:
    u_prd_tau['direct'][tau] = {'resolution': resolution}
    u_prd_tau['direct'][tau]['u'] = _get_estimations_in_batches(
      direct=True,
      resolution=resolution,
      tau=tau,
      transform=(lambda arr: change_resolution(arr, resolution)),
    )

  # Autoregressive rollout
  resolution = resolution_train
  for tau_max in taus_rollout:
    u_prd_tau['rollout'][tau_max] = {'resolution': resolution}
    u_prd_tau['rollout'][tau_max]['u'] = _get_estimations_in_batches(
      direct=False,
      resolution=resolution,
      tau=tau_max,
      transform=(lambda arr: change_resolution(arr, resolution)),
    )

  # Spatial continuity
  tau_max = train_flags['time_downsample_factor'] * train_flags['direct_steps']
  tau = train_flags['time_downsample_factor']
  for resolution in resolutions:
    u_prd_px['direct'][resolution] = {'resolution': resolution}
    u_prd_px['direct'][resolution]['u'] = _get_estimations_in_batches(
      direct=True,
      resolution=resolution,
      tau=tau,
      transform=(lambda arr: change_resolution(arr, resolution)),
    )
    u_prd_px['rollout'][resolution] = {'resolution': resolution}
    u_prd_px['rollout'][resolution]['u'] = _get_estimations_in_batches(
      direct=False,
      resolution=resolution,
      tau=tau_max,
      transform=(lambda arr: change_resolution(arr, resolution)),
    )

  # Noise control
  tau_max = train_flags['time_downsample_factor'] * train_flags['direct_steps']
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
    u_prd_noise['direct'][noise_level] = {'resolution': resolution}
    u_prd_noise['direct'][noise_level]['u'] = _get_estimations_in_batches(
      direct=True,
      resolution=resolution,
      tau=tau,
      transform=transform,
    )
    # Rollout estimations
    u_prd_noise['rollout'][noise_level] = {'resolution': resolution}
    u_prd_noise['rollout'][noise_level]['u'] = _get_estimations_in_batches(
      direct=False,
      resolution=resolution,
      tau=tau_max,
      transform=transform,
    )

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

  @functools.partial(jax.pmap, static_broadcasted_argnums=(0,))
  def _get_rollout_estimations(
    num_steps: int,
    variables,
    stats,
    u_inp: Array,
    key = None,
  ):
    """Inputs are of shape [batch_size_per_device, ...]"""

    batch_size = u_inp.shape[0]
    rollout, _ = unroller.unroll(
      variables,
      stats,
      u_inp,
      jnp.array([0.]).repeat(repeats=(batch_size)).reshape(batch_size, 1),
      num_steps,
      key,
    )

    return rollout

  def _get_estimations_in_batches(
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

  # Autoregressive rollout
  u_prd = []
  for _ in range(repeats):
    subkey, key = jax.random.split(key)
    u_prd.append(
      _get_estimations_in_batches(
        transform=(lambda arr: change_resolution(arr, resolution_train)),
        key=subkey,
      )
    )
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
  direct_steps = configs['flags']['direct_steps']
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
  tau_max = time_downsample_factor * (direct_steps + (direct_steps > 1))
  taus_direct = list(range(tau_min, tau_max + 1))
  if not interpolate_tau:
    taus_direct = [tau for tau in taus_direct if (tau % 2) == 0]
  # NOTE: One compilation per tau_rollout
  taus_rollout = [time_downsample_factor * d for d in range(1, direct_steps+1)]
  # NOTE: Two compilations per resolution
  resolutions = [(px, px) for px in [32, 48, 64, 96, 128]]
  noise_levels = [0, .002, .005, .01]

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
  def _get_err_trajectory(_u_gtr, _u_prd):
    _err = [
      np.median(rel_lp_error_norm(_u_gtr[:, [idx_t]], _u_prd[:, [idx_t]], p=1)).item() * 100
      for idx_t in range(_u_gtr.shape[1])
    ]
    return _err
  errors = {
    key: {
      'direct': {
        str(subkey): _get_err_trajectory(
          _u_gtr=change_resolution(u_gtr, resolution=u_prd[key]['direct'][subkey]['resolution']),
          _u_prd=u_prd[key]['direct'][subkey]['u'],
        )
        for subkey in u_prd[key]['direct'].keys()
      },
      'rollout': {
        str(subkey): _get_err_trajectory(
          _u_gtr=change_resolution(u_gtr, resolution=u_prd[key]['rollout'][subkey]['resolution']),
          _u_prd=u_prd[key]['rollout'][subkey]['u'],
        )
        for subkey in u_prd[key]['rollout'].keys()
      },
    }
    for key in u_prd.keys()
  }

  with open(DIR_TESTS / 'errors.json', 'w') as f:
    json.dump(obj=errors, fp=f)

  # Plot the errors and store the plots
  (DIR_FIGS / 'errors').mkdir()
  def errors_to_df(_errors):
    df = pd.DataFrame(_errors)
    df['t'] = df.index
    df = df.melt(id_vars=['t'], value_name='error')
    return df
  # Temporal continuity
  g = plot_error_vs_time(
    df=errors_to_df(errors['tau']['direct']),
    idx_fn=IDX_FN,
    variable_title='$\\tau$',
  )
  g.figure.savefig(DIR_FIGS / 'errors' / 'tau-direct.png')
  plt.close(g.figure)
  g = plot_error_vs_time(
    df=errors_to_df(errors['tau']['rollout']),
    idx_fn=IDX_FN,
    variable_title='$\\tau_{max}$',
  )
  g.figure.savefig(DIR_FIGS / 'errors' / 'tau-rollout.png')
  plt.close(g.figure)
  # Noise control
  g = plot_error_vs_time(
    df=errors_to_df(errors['noise']['direct']),
    idx_fn=IDX_FN,
    variable_title='$\\dfrac{\\sigma_{noise}}{\\sigma_{data}}$',
  )
  g.figure.savefig(DIR_FIGS / 'errors' / 'noise-direct.png')
  plt.close(g.figure)
  g = plot_error_vs_time(
    df=errors_to_df(errors['noise']['rollout']),
    idx_fn=IDX_FN,
    variable_title='$\\dfrac{\\sigma_{noise}}{\\sigma_{data}}$',
  )
  g.figure.savefig(DIR_FIGS / 'errors' / 'noise-rollout.png')
  plt.close(g.figure)
  # Spatial continuity
  g = plot_error_vs_time(
    df=errors_to_df(errors['px']['direct']),
    idx_fn=IDX_FN,
    variable_title='Resolution',
  )
  g.figure.savefig(DIR_FIGS / 'errors' / 'resolution-direct.png')
  plt.close(g.figure)
  g = plot_error_vs_time(
    df=errors_to_df(errors['px']['rollout']),
    idx_fn=IDX_FN,
    variable_title='Resolution',
  )
  g.figure.savefig(DIR_FIGS / 'errors' / 'resolution-rollout.png')
  plt.close(g.figure)

  # Get ensemble estimations with the default settings
  # NOTE: One compilation
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

  logging.info('done')

if __name__ == '__main__':
  logging.set_verbosity('info')
  app.run(main)
