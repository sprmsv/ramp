import json
import pickle
import shutil
from time import time
from typing import Tuple, Type, Mapping, Callable, Any, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint
import pandas as pd
import scipy.ndimage
from absl import app, flags, logging
from flax.training.common_utils import shard, shard_prng_key
from flax.jax_utils import replicate

from rigno.dataset import Dataset, Batch
from rigno.experiments import DIR_EXPERIMENTS
from rigno.metrics import rel_lp_error_norm
from rigno.models.operator import AbstractOperator, Inputs
from rigno.models.rigno import RIGNO, RegionInteractionGraphs, RegionInteractionGraphBuilder
from rigno.models.unet import UNet
from rigno.plot import plot_estimates, plot_ensemble, plot_error_vs_time
from rigno.stepping import Stepper, TimeDerivativeStepper, ResidualStepper, OutputStepper
from rigno.stepping import AutoregressiveStepper
from rigno.utils import Array, disable_logging, profile


NUM_DEVICES = jax.local_device_count()
IDX_FN = 14

FLAGS = flags.FLAGS

def define_flags():
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

# TMP Finish up
def change_discretization(u: Array):
  ...

def profile_inferrence(
  dataset: Dataset,
  graphs: RegionInteractionGraphs,
  model: AbstractOperator,
  stepping: Type[Stepper],
  state: dict,
  stats: dict,
  p_edge_masking: float,
  repeats: int = 10,
  jit: bool = True,
):

  # Configure and build new model
  model_configs = model.configs
  if isinstance(model, RIGNO):
    model_configs['p_dropout_edges_grid2mesh'] = p_edge_masking
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

  # Set model inputs
  model_kwargs = dict(
    variables={'params': state['params']},
    stats=stats,
    inputs=Inputs(
      u=batch.u[:, [0]],
      c=(batch.c[:, [0]] if (batch.c is not None) else None),
      x_inp=batch._x,
      x_out=batch._x,
      t=batch.t[:, [0]],
      tau=dataset.dt,
    ),
    graphs=graphs,
  )

  # Profile compilation
  t_compilation = profile(f=apply_fn, kwargs=model_kwargs, repeats=1)
  # Profile inferrence after compilation
  t = profile(f=apply_fn, kwargs=model_kwargs, repeats=repeats) / repeats

  general_info = [
    'NUMBER OF DEVICES: 1',
    f'BATCH SIZE PER DEVICE: {batch_size_per_device}',
    f'MODEL: {model.__class__.__name__}',
    f'p_edge_masking: {p_edge_masking}',
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

# TMP Update
def get_direct_estimations(
  step: Stepper.apply,
  variables,
  stats,
  graphs: RegionInteractionGraphs,
  batch: Batch,
  tau: float,
  time_downsample_factor: int = 1,  # TMP REMOVE?
  key = None,
) -> Array:
  """Inputs are of shape [batch_size_per_device, ...]"""

  # Set lead times
  lead_times = jnp.arange(batch.shape[1])
  batch_size = batch.shape[0]

  # Get inputs for all lead times
  # -> [num_lead_times, batch_size_per_device, ...]
  u_inp = jax.vmap(
      lambda lt: jax.lax.dynamic_slice_in_dim(
        operand=batch.u,
        start_index=(lt), slice_size=1, axis=1)
  )(lead_times)
  t_inp = batch.t.swapaxes(0, 1).reshape(-1, batch_size, 1)

  # Get model estimations
  def _use_step_on_mini_batches(carry, x):
    idx = carry
    inputs = Inputs(
      u=u_inp[idx],
      c=(batch.c[:, [idx]] if (batch.c is not None) else None),
      x_inp=batch._x,
      x_out=batch._x,
      t=(t_inp[idx] / time_downsample_factor),
      tau=tau,
    )
    _u_prd = step(
      variables=variables,
      stats=stats,
      inputs=inputs,
      graphs=graphs,
      key=key,
    )
    carry += 1
    return carry, _u_prd
  # -> [num_lead_times, batch_size_per_device, 1, ...]
  _, u_prd = jax.lax.scan(
    f=_use_step_on_mini_batches,
    init=0,
    xs=None,
    length=batch.shape[1],
  )

  # Re-arrange
  # -> [batch_size_per_device, num_lead_times, ...]
  u_prd = u_prd.swapaxes(0, 1).squeeze(axis=2)

  return u_prd

# TMP Update
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

# TMP Update
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
    if isinstance(model, RIGNO):
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
        dt=(1. / train_flags['time_downsample_factor'])
      )
      apply_unroll_jit[resolution][tau_max] = jax.jit(
        unrollers[resolution][tau_max].unroll, static_argnums=(4,))

  # Set the groundtruth solutions
  u_gtr = next(dataset.batches(mode='test', batch_size=dataset.nums['test']))

  # Instantiate the outputs
  errors = {error_type: {
      key: {'direct': {}, 'rollout': {}}
      for key in ['tau', 'px', 'noise']
    }
    for error_type in ['l1', 'l2']
  }
  u_prd_rollout = None

  # Define a auxiliary function for getting the median of the errors
  def _get_err_trajectory(_u_gtr, _u_prd, p):
    _err = [
      np.median(rel_lp_error_norm(_u_gtr[:, [idx_t]], _u_prd[:, [idx_t]], p=p)).item() * 100
      for idx_t in range(_u_gtr.shape[1])
    ]
    return _err

  # Temporal continuity
  resolution = resolution_train
  for tau in taus_direct:
    if tau == .5:
      _apply_stepper = apply_steppers_twice_jit[resolution]
    else:
      _apply_stepper = apply_steppers_jit[resolution]
    t0 = time()
    u_prd = _get_estimations_in_batches(
      direct=True,
      apply_fn=_apply_stepper,
      tau=(tau if tau != .5 else 1),
      transform=(lambda arr: change_resolution(arr, resolution)),
    )
    errors['l1']['tau']['direct'][tau] = _get_err_trajectory(
      _u_gtr=change_resolution(u_gtr, resolution=resolution),
      _u_prd=u_prd,
      p=1,
    )
    errors['l2']['tau']['direct'][tau] = _get_err_trajectory(
      _u_gtr=change_resolution(u_gtr, resolution=resolution),
      _u_prd=u_prd,
      p=2,
    )
    del u_prd
    print_between_dashes(f'tau_direct={tau} \t TIME={time()-t0 : .4f}s')

  # Autoregressive rollout
  resolution = resolution_train
  for tau_max in taus_rollout:
    t0 = time()
    u_prd = _get_estimations_in_batches(
      direct=False,
      apply_fn=apply_unroll_jit[resolution][tau_max],
      transform=(lambda arr: change_resolution(arr, resolution)),
    )
    errors['l1']['tau']['rollout'][tau_max] = _get_err_trajectory(
      _u_gtr=change_resolution(u_gtr, resolution=resolution),
      _u_prd=u_prd,
      p=1,
    )
    errors['l2']['tau']['rollout'][tau_max] = _get_err_trajectory(
      _u_gtr=change_resolution(u_gtr, resolution=resolution),
      _u_prd=u_prd,
      p=2,
    )
    if tau_max == train_flags['time_downsample_factor']:
      u_prd_rollout = u_prd[:, [0, IDX_FN, -1]]
    del u_prd
    print_between_dashes(f'tau_max={tau_max} \t TIME={time()-t0 : .4f}s')

  # Spatial continuity
  tau_max = train_flags['time_downsample_factor']
  tau = train_flags['time_downsample_factor']
  for resolution in resolutions:
    # Direct
    t0 = time()
    u_prd = _get_estimations_in_batches(
      direct=True,
      apply_fn=apply_steppers_jit[resolution],
      tau=tau,
      transform=(lambda arr: change_resolution(arr, resolution)),
    )
    errors['l1']['px']['direct'][resolution] = _get_err_trajectory(
      _u_gtr=change_resolution(u_gtr, resolution=resolution),
      _u_prd=u_prd,
      p=1,
    )
    errors['l2']['px']['direct'][resolution] = _get_err_trajectory(
      _u_gtr=change_resolution(u_gtr, resolution=resolution),
      _u_prd=u_prd,
      p=2,
    )
    del u_prd
    print_between_dashes(f'resolution={resolution} (direct) \t TIME={time()-t0 : .4f}s')
    # Rollout
    t0 = time()
    u_prd = _get_estimations_in_batches(
      direct=False,
      apply_fn=apply_unroll_jit[resolution][tau_max],
      transform=(lambda arr: change_resolution(arr, resolution)),
    )
    errors['l1']['px']['rollout'][resolution] = _get_err_trajectory(
      _u_gtr=change_resolution(u_gtr, resolution=resolution),
      _u_prd=u_prd,
      p=1,
    )
    errors['l2']['px']['rollout'][resolution] = _get_err_trajectory(
      _u_gtr=change_resolution(u_gtr, resolution=resolution),
      _u_prd=u_prd,
      p=2,
    )
    del u_prd
    print_between_dashes(f'resolution={resolution} (rollout) \t TIME={time()-t0 : .4f}s')

  # Noise control
  tau_max = train_flags['time_downsample_factor']
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
    u_prd = _get_estimations_in_batches(
      direct=True,
      apply_fn=apply_steppers_jit[resolution],
      tau=tau,
      transform=transform,
    )
    errors['l1']['noise']['direct'][noise_level] = _get_err_trajectory(
      _u_gtr=change_resolution(u_gtr, resolution=resolution),
      _u_prd=u_prd,
      p=1,
    )
    errors['l2']['noise']['direct'][noise_level] = _get_err_trajectory(
      _u_gtr=change_resolution(u_gtr, resolution=resolution),
      _u_prd=u_prd,
      p=2,
    )
    del u_prd
    print_between_dashes(f'noise_level={noise_level} (direct) \t TIME={time()-t0 : .4f}s')
    # Rollout estimations
    t0 = time()
    u_prd = _get_estimations_in_batches(
      direct=False,
      apply_fn=apply_unroll_jit[resolution][tau_max],
      transform=transform,
    )
    errors['l1']['noise']['rollout'][noise_level] = _get_err_trajectory(
      _u_gtr=change_resolution(u_gtr, resolution=resolution),
      _u_prd=u_prd,
      p=1,
    )
    errors['l2']['noise']['rollout'][noise_level] = _get_err_trajectory(
      _u_gtr=change_resolution(u_gtr, resolution=resolution),
      _u_prd=u_prd,
      p=2,
    )
    del u_prd
    print_between_dashes(f'noise_level={noise_level} (rollout) \t TIME={time()-t0 : .4f}s')

  return errors, u_prd_rollout

# TMP Update
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
  if isinstance(model, RIGNO):
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
    dt=(1. / train_flags['time_downsample_factor'])
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
      )[:, [0, IDX_FN, -1]]
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
    include_passive_variables=False,
    concatenate_coeffs=False,
    time_downsample_factor=1,
    space_downsample_factor=1,
    unstructured=False,
    n_train=0,
    n_valid=0,
    n_test=FLAGS.n_test,
    preload=True,
  )
  dataset_small = Dataset(
    datadir=FLAGS.datadir,
    datapath=datapath,
    include_passive_variables=False,
    concatenate_coeffs=False,
    time_downsample_factor=1,
    space_downsample_factor=1,
    unstructured=False,
    n_train=0,
    n_valid=0,
    n_test=min(4, FLAGS.n_test),
    preload=True,
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
  model_configs = configs['model_configs']
  resolution_train = tuple(configs['resolution'])
  # Read the state
  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  mngr = orbax.checkpoint.CheckpointManager(DIR / 'checkpoints')
  best_checkpointed_step = mngr.best_step()
  ckpt = orbax_checkpointer.restore(directory=(DIR / 'checkpoints' / str(best_checkpointed_step) / 'default'))
  state = jax.tree_util.tree_map(jnp.array, ckpt['state'])

  # Set the stepper type
  if configs['flags']['stepper'] == 'out':
    stepping = OutputStepper
  elif configs['flags']['stepper'] == 'res':
    stepping = ResidualStepper
  elif configs['flags']['stepper'] == 'der':
    stepping = TimeDerivativeStepper
  else:
    raise ValueError

  # Set the model
  model = RIGNO(**model_configs)

  # Define the graph builder
  builder = RegionInteractionGraphBuilder(
    periodic=dataset.metadata.periodic,
    rmesh_levels=configs['flags']['rmesh_levels'],
    subsample_factor=configs['flags']['mesh_subsample_factor'],
    overlap_factor_p2r=configs['flags']['overlap_factor_p2r'],
    overlap_factor_r2p=configs['flags']['overlap_factor_r2p'],
    node_coordinate_freqs=configs['flags']['node_coordinate_freqs'],
  )

  # TMP TODO: Build a graph for each discretization
  # Construct the graphs
  # NOTE: Assuming fix mesh for all batches
  graphs = builder.build(
    x_inp=dataset.sample._x,
    x_out=dataset.sample._x,
    domain=np.array(dataset.metadata.domain_x),
    key=None,
  )

  # Profile
  # NOTE: One compilation
  if FLAGS.profile:
    profile_inferrence(
      dataset=dataset,
      graphs=graphs,
      model=model,
      stepping=stepping,
      state=state,
      stats=stats,
      p_edge_masking=.0,
    )

  # Create a clean directory for tests
  DIR_TESTS = DIR / 'tests'
  DIR_FIGS = DIR_TESTS / 'figures'
  if DIR_TESTS.exists():
    shutil.rmtree(DIR_TESTS)
  DIR_TESTS.mkdir()
  DIR_FIGS.mkdir()

  # Set evaluation settings  # TMP
  tau_min = 1
  tau_max = (tau_max_train + (tau_max_train > 1))
  taus_direct = [.5] + list(range(tau_min, tau_max + 1))
  # NOTE: One compilation per tau_rollout
  taus_rollout = [.5, 1] + [time_downsample_factor * d for d in range(1, tau_max_train+1)]
  # NOTE: Two compilations per resolution
  space_downsample_factors = [4, 3, 2, 1.5, 1] if FLAGS.resolution else []
  noise_levels = [0, .005, .01, .02] if FLAGS.noise else []

  # Set the groundtruth trajectories
  u_gtr = next(dataset.batches(mode='test', batch_size=dataset.nums['test']))
  u_gtr_small = next(dataset_small.batches(mode='test', batch_size=dataset_small.nums['test']))

  # Get model estimations with all settings
  errors, u_prd = get_all_estimations(
    dataset=dataset,
    model=model,
    stepping=stepping,
    state=state,
    stats=stats,
    resolution_train=resolution_train,
    train_flags=configs['flags'],
    taus_direct=taus_direct,
    taus_rollout=taus_rollout,
    resolutions=space_downsample_factors,
    noise_levels=noise_levels,
    p_edge_masking_grid2mesh=0,
  )

  # Plot estimation visualizations
  (DIR_FIGS / 'samples').mkdir()
  for s in range(min(4, FLAGS.n_test)):
    fig, _ = plot_estimations(
      u_gtr=change_resolution(u_gtr[:, [0, IDX_FN, -1]], resolution_train),
      u_prd=u_prd,
      idx_out=1,
      idx_inp=0,
      idx_traj=s,
      symmetric=dataset.metadata.signed,
      names=dataset.metadata.names,
    )
    fig.savefig(DIR_FIGS / 'samples' / f'rollout-fn-s{s:02d}.png')
    plt.close(fig)
    fig, _ = plot_estimations(
      u_gtr=change_resolution(u_gtr[:, [0, IDX_FN, -1]], resolution_train),
      u_prd=u_prd,
      idx_out=2,
      idx_inp=0,
      idx_traj=s,
      symmetric=dataset.metadata.signed,
      names=dataset.metadata.names,
    )
    fig.savefig(DIR_FIGS / 'samples' / f'rollout-ex-s{s:02d}.png')
    plt.close(fig)

  # Store the errors
  for error_type in errors.keys():
    for key in errors[error_type]['px'].keys():
      errors[error_type]['px'][key] = {
        str(resolution): errors[error_type]['px'][key][resolution]
        for resolution in errors[error_type]['px'][key].keys()
      }
  with open(DIR_TESTS / 'errors.json', 'w') as f:
    json.dump(obj=errors, fp=f)

  # Print minimum errors
  l1_final = min([errors['l1']['tau']['rollout'][tau][IDX_FN] for tau in taus_rollout])
  l2_final = min([errors['l2']['tau']['rollout'][tau][IDX_FN] for tau in taus_rollout])
  l1_extra = min([errors['l2']['tau']['rollout'][tau][-1] for tau in taus_rollout])
  l2_extra = min([errors['l2']['tau']['rollout'][tau][-1] for tau in taus_rollout])
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
  errors_plot = errors['l1']
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
      tau_max=time_downsample_factor,
      p_edge_masking_grid2mesh=0.5,
      key=subkey,
    )

    # Plot ensemble statistics
    (DIR_FIGS / 'ensemble').mkdir()
    for s in range(min(4, FLAGS.n_test)):
      fig, _ = plot_ensemble(
        u_gtr=change_resolution(u_gtr_small[:, [0, IDX_FN, -1]], resolution_train),
        u_ens=u_prd_ensemble,
        idx_out=1,
        idx_traj=s,
        symmetric=dataset_small.metadata.signed,
        names=dataset_small.metadata.names,
      )
      fig.savefig(DIR_FIGS / 'ensemble' / f'rollout-fn-s{s:02d}.png')
      plt.close(fig)
      fig, _ = plot_ensemble(
        u_gtr=change_resolution(u_gtr_small[:, [0, IDX_FN, -1]], resolution_train),
        u_ens=u_prd_ensemble,
        idx_out=2,
        idx_traj=s,
        symmetric=dataset_small.metadata.signed,
        names=dataset_small.metadata.names,
      )
      fig.savefig(DIR_FIGS / 'ensemble' / f'rollout-ex-s{s:02d}.png')
      plt.close(fig)

  print_between_dashes('DONE')

if __name__ == '__main__':
  logging.set_verbosity('info')
  define_flags()
  app.run(main)
