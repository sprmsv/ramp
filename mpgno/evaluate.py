import functools
import pickle
import json
from pathlib import Path
from typing import Tuple

import jax
import numpy as np
import jax.numpy as jnp
import orbax.checkpoint
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from absl import app, flags, logging
from flax.training.common_utils import shard
from flax.jax_utils import replicate

from mpgno.models.mpgno import AbstractOperator, MPGNO
from mpgno.models.mpgno import MPGNO
from mpgno.utils import Array, disable_logging, profile
from mpgno.stepping import Stepper, TimeDerivativeUpdater, ResidualUpdater, OutputUpdater
from mpgno.stepping import AutoregressiveStepper
from mpgno.experiments import DIR_EXPERIMENTS
from mpgno.dataset import Dataset
from mpgno.metrics import rel_lp_error
from mpgno.plot import CMAP_BBR, CMAP_BWR, CMAP_WRB
from mpgno.plot import plot_trajectory, plot_estimations, plot_ensemble


NUM_DEVICES = jax.local_device_count()
IDX_FN = 14

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
flags.DEFINE_integer(name='batch_size', default=4, required=False,
  help='Size of a batch of training samples'
)
flags.DEFINE_integer(name='n_test', default=(2**8), required=False,
  help='Number of test samples'
)

# TODO: Compute median errors and store them
# TODO: Plot and store the plots systematically

def change_resolution(u: Array, resolution: Tuple[int, int]):
  # TODO: Use interpolation
  space_downsample_factor = [
    (res_before // resolution)
    for (res_before, resolution) in zip(u.shape[2:4], resolution)
  ]

  return u[:, :, ::space_downsample_factor[0], ::space_downsample_factor[1]]

def profile_inferrence(
  dataset: Dataset,
  stepper: Stepper,
  state,
  stats,
  tau,
  resolution: Tuple[int, int],
  time_downsample_factor: int,
):

  # Set the steppers
  predictor = AutoregressiveStepper(
    stepper=stepper,
    tau_max=(tau / time_downsample_factor),
    tau_base=(1. / time_downsample_factor)
  )

  # Get a batch and transform it
  batch_size_per_device = FLAGS.batch_size // NUM_DEVICES
  batch = next(dataset.batches(mode='test', batch_size=batch_size_per_device))
  num_times = batch.shape[1]
  u_inp = batch[:, [0]]
  u_inp = change_resolution(u_inp, resolution)

  # Jit and set the input arguments
  unroll_jit = jax.jit(predictor.unroll, static_argnames=('num_steps',))
  unroll_kwargs = dict(
    variables={'params': state['params']},
    stats=stats,
    u_inp=u_inp,
    t_inp=jnp.array([0.]).repeat(repeats=(batch_size_per_device)).reshape(batch_size_per_device, 1),
    num_steps=num_times,
  )

  # Profile with and without jit
  t_jit_compilation = profile(unroll_jit, unroll_kwargs)
  t_jit = profile(unroll_jit, unroll_kwargs, repeats=10) / 10
  t_raw = profile(predictor.unroll, unroll_kwargs, repeats=10) / 10

  # Print the results
  print('-' * 100)
  print('-' * 50)
  print('General information')
  print('-' * 50)
  print(f'Number of devices: {NUM_DEVICES}')
  print(f'n_test: {FLAGS.n_test}')
  print(f'batch_size: {FLAGS.batch_size}')
  print(f'batch_size_per_device: {batch_size_per_device}')
  print(f'resolution: {resolution}')
  print(f'tau: {tau}')
  print('-' * 50)
  print('Time per trajectory')
  print('-' * 50)
  print(f'unroll_jit (compilation): {t_jit_compilation / batch_size_per_device : .4f}s')
  print(f'unroll_jit (inferrence): {t_jit / batch_size_per_device : .4f}s')
  print(f'unroll_raw (inferrence): {t_raw / batch_size_per_device : .4f}s')
  print('-' * 50)
  minimum_efficient_batches = t_jit_compilation / (t_raw - t_jit)
  print(f'Test samples to reach the compiled time: {minimum_efficient_batches * batch_size_per_device : .1f}')
  n_batches = FLAGS.n_test // FLAGS.batch_size
  print(f'Total time for n_test (with jit) : {(t_jit_compilation + (n_batches-1) * t_jit) : .1f}')
  print(f'Total time for n_test (without jit) : {(t_raw * n_batches) : .1f}')

def get_estimations(
  direct: bool,
  dataset: Dataset,
  model: AbstractOperator,
  stepping,
  state,
  stats,
  tau,
  resolution: Tuple[int, int],
  time_downsample_factor: int,
  noise_level: float = None,
  p_edge_masking_grid2mesh: float = .0,
):

  @functools.partial(jax.pmap, static_broadcasted_argnums=(0,))
  def get_direct_estimations(
    stepper: Stepper,
    variables,
    stats,
    trajs: Array,
    tau: int,
    time_downsample_factor: int,
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
    t_inp = lead_times.repeat(repeats=batch_size).reshape(-1, batch_size)

    # Re-align
    # -> [batch_size_per_device * num_lead_times, ...]
    u_inp = u_inp.swapaxes(0, 1).reshape(-1, 1, *trajs.shape[2:])
    t_inp = t_inp.swapaxes(0, 1).reshape(-1, 1)

    # Get model estimations
    # -> [batch_size_per_device * num_lead_times, ...]
    u_prd = stepper.apply(
      variables=variables,
      stats=stats,
      u_inp=u_inp,
      t_inp=(t_inp / time_downsample_factor),
      tau=(tau / time_downsample_factor),
    )

    # Re-align
    # -> [batch_size_per_device, num_lead_times, ...]
    u_prd = u_prd.reshape(*trajs.shape)

    return u_prd

  @functools.partial(jax.pmap, static_broadcasted_argnums=(0, 1,))
  def get_rollout_estimations(
    predictor: AutoregressiveStepper,
    num_steps: int,
    variables,
    stats,
    u_inp: Array,
  ):
    """Inputs are of shape [batch_size_per_device, ...]"""

    batch_size = u_inp.shape[0]
    rollout, _ = predictor.unroll(
      variables=variables,
      stats=stats,
      u_inp=u_inp,
      t_inp=jnp.array([0.]).repeat(repeats=(batch_size)).reshape(batch_size, 1),
      num_steps=num_steps,
    )

    return rollout

  # Configure and build new model
  configs = model.configs
  if isinstance(model, MPGNO):
    configs['num_grid_nodes'] = resolution
    configs['p_dropout_edges_grid2mesh'] = p_edge_masking_grid2mesh
    configs['p_dropout_edges_multimesh'] = 0.
    configs['p_dropout_edges_mesh2grid'] = 0.
  else:
    raise NotImplementedError
  model = model.__class__(**configs)

  # Set the stepper
  stepper = stepping(operator=model)
  predictor = AutoregressiveStepper(
    stepper=stepper,
    tau_max=(tau / time_downsample_factor),
    tau_base=(1. / time_downsample_factor)
  )

  batches = dataset.batches(mode='test', batch_size=FLAGS.batch_size)

  # Replicate state and stats
  # NOTE: Internally uses jax.device_put_replicate
  state = replicate(state)
  stats = replicate(stats)

  # Loop over the batches
  u_prd = []
  for batch in batches:
    # Change the resolution
    batch = change_resolution(batch, resolution)

    # Add noise
    if noise_level:
      batch_std = np.std(batch, axis=(0, 2, 3), keepdims=True)
      batch += noise_level * np.random.normal(scale=batch_std, size=batch.shape)

    # Split the batch between devices
    # -> [NUM_DEVICES, batch_size_per_device, ...]
    batch = shard(batch)

    # Get the direct predictions
    if direct:
      u_prd_batch = get_direct_estimations(
        stepper,
        variables={'params': state['params']},
        stats=stats,
        trajs=batch,
        tau=replicate(tau),
        time_downsample_factor=replicate(time_downsample_factor),
      )
      # Replace the tail of the predictions with the head of the input
      u_prd_batch = jnp.concatenate([batch[:, :, :tau], u_prd_batch[:, :, :-tau]], axis=2)

    # Get the rollout predictions
    else:
      num_times = batch.shape[2]
      u_prd_batch = get_rollout_estimations(
        predictor,
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

def get_all_estimations(
  dataset,
  model,
  stepping,
  state,
  stats,
  resolution_train,
  flags,
):

  # Instantiate the outputs
  u_prd_tau = {'direct': {}, 'rollout': {}}
  u_prd_px = {'direct': {}, 'rollout': {}}
  u_prd_noise = {'direct': {}, 'rollout': {}}

  # Set taus
  extrapolate_tau_dr = 1
  interpolate_tau_dr = True
  tau_min = 1
  tau_max = flags['time_downsample_factor'] * (flags['direct_steps'] + extrapolate_tau_dr)
  taus_dr = list(range(tau_min, tau_max + 1))
  if not interpolate_tau_dr:
    taus_dr = [tau for tau in taus_dr if (tau % 2) == 0]

  # Time continuity
  for tau in taus_dr:
    u_prd_tau['direct'][tau] = get_estimations(
      direct=True,
      dataset=dataset,
      model=model,
      stepping=stepping,
      state=state,
      stats=stats,
      tau=tau,
      noise_level=0,
      p_edge_masking_grid2mesh=0,
      resolution=resolution_train,
      time_downsample_factor=flags['time_downsample_factor'],
    )

  # Autoregressive rollout
  for tau in [2, 4, 8]:
    u_prd_tau['rollout'][tau] = get_estimations(
      direct=False,
      dataset=dataset,
      model=model,
      stepping=stepping,
      state=state,
      stats=stats,
      tau=tau,
      noise_level=0,
      p_edge_masking_grid2mesh=0,
      resolution=resolution_train,
      time_downsample_factor=flags['time_downsample_factor'],
    )

  # Spatial resolution
  # TODO: Add other resolutions (use scipy.interpolate.RegularGridInterpolator)
  for px in [32, 64, 128]:
    u_prd_px['direct'][px] = get_estimations(
      direct=True,
      dataset=dataset,
      model=model,
      stepping=stepping,
      state=state,
      stats=stats,
      tau=flags['time_downsample_factor'],
      noise_level=0,
      p_edge_masking_grid2mesh=0,
      resolution=(px, px),
      time_downsample_factor=flags['time_downsample_factor'],
    )
    u_prd_px['rollout'][px] = get_estimations(
      direct=False,
      dataset=dataset,
      model=model,
      stepping=stepping,
      state=state,
      stats=stats,
      tau=flags['time_downsample_factor'],
      noise_level=0,
      p_edge_masking_grid2mesh=0,
      resolution=(px, px),
      time_downsample_factor=flags['time_downsample_factor'],
    )

  # Noise control
  for level in [0, .005, .01, .05, .1]:
    u_prd_noise['direct'][level] = get_estimations(
      direct=True,
      dataset=dataset,
      model=model,
      stepping=stepping,
      state=state,
      stats=stats,
      tau=flags['time_downsample_factor'],
      noise_level=level,
      p_edge_masking_grid2mesh=0,
      resolution=resolution_train,
      time_downsample_factor=flags['time_downsample_factor'],
    )
    u_prd_noise['rollout'][level] = get_estimations(
      direct=False,
      dataset=dataset,
      model=model,
      stepping=stepping,
      state=state,
      stats=stats,
      tau=flags['time_downsample_factor'],
      noise_level=level,
      p_edge_masking_grid2mesh=0,
      resolution=resolution_train,
      time_downsample_factor=flags['time_downsample_factor'],
    )

  return u_prd_tau, u_prd_px, u_prd_noise

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

  assert FLAGS.batch_size % NUM_DEVICES == 0

  dataset = Dataset(
    datadir=FLAGS.datadir,
    datapath=FLAGS.datapath,
    n_train=0,
    n_valid=0,
    n_test=FLAGS.n_test,
    preload=True,
    time_downsample_factor=1,
    space_downsample_factor=1,
  )

  experiments = (DIR_EXPERIMENTS / f'E{FLAGS.exp}' / FLAGS.datapath).iterdir()
  D = next(experiments)  # TMP

  # Read the stats
  with open(D / 'stats.pkl', 'rb') as f:
    stats = pickle.load(f)
  stats = {
      key: {k: jnp.array(v) for k, v in val.items()}
      for key, val in stats.items()
    }

  # Read the configs
  with open(D / 'configs.json', 'rb') as f:
    configs = json.load(f)
  model_configs = configs['model_configs']
  resolution_train = model_configs['num_grid_nodes']

  # Read the state
  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  mngr = orbax.checkpoint.CheckpointManager(D / 'checkpoints')
  best_checkpointed_step = mngr.best_step()
  ckpt = orbax_checkpointer.restore(directory=(D / 'checkpoints' / str(best_checkpointed_step) / 'default'))
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
  model = MPGNO(**model_configs)
  stepper = stepping(operator=model)

  # Profile
  profile_inferrence(
    dataset=dataset,
    stepper=stepper,
    state=state,
    stats=stats,
    tau=configs['flags']['time_downsample_factor'],
    resolution=resolution_train,
    time_downsample_factor=configs['flags']['time_downsample_factor'],
  )

  # _ = get_all_estimations(
  #   dataset=dataset,
  #   model=model,
  #   stepping=stepping,
  #   state=state,
  #   stats=stats,
  #   resolution_train=resolution_train,
  #   flags=configs['flags'],
  # )

  # u_gtr = next(dataset.batches(mode='test', batch_size=dataset.nums['test']))
  # u_gtr = change_resolution(u_gtr, resolution)

  # err = rel_lp_error(u_gtr, u_prd)
  # print(err)


if __name__ == '__main__':
  logging.set_verbosity('info')
  app.run(main)
