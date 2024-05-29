"""Utility functions for reading the datasets."""

from pathlib import Path
import h5py
import numpy as np
from typing import Any, Union, Sequence
from dataclasses import dataclass

import numpy as np
import jax
import jax.lax
import jax.numpy as jnp
import flax.typing

from mpgno.utils import Array
from mpgno.utils import normalize
from mpgno.models.utils import compute_derivatives


@dataclass
class Metadata:
  data_group: str
  active_variables: Sequence[int]
  target_variables: Sequence[int]
  stats: dict[str, Sequence[float]]
  signed: Union[bool, Sequence[bool]] = True

  @property
  def stats_target_variables(self) -> dict[str, np.array]:
    _stats = {
      'mean': np.array(self.stats['mean']).reshape(1, 1, 1, 1, -1)[..., self.target_variables],
      'std': np.array(self.stats['std']).reshape(1, 1, 1, 1, -1)[..., self.target_variables],
    }
    return _stats

ACTIVE_VARS_INCOMPRESSIBLE_FLUIDS = [0, 1]
ACTIVE_VARS_COMPRESSIBLE_FLOW = [0, 1, 2, 3]

TARGET_VARS_INCOMPRESSIBLE_FLUIDS = [0, 1]
TARGET_VARS_COMPRESSIBLE_FLOW = [1, 2]

STATS_INCOMPRESSIBLE_FLUIDS = {
  'mean': [0., 0.],
  'std': [.391, .356],
}
STATS_COMPRESSIBLE_FLOW = {
  'mean': [.80, 0., 0., .553, None],
  'std': [.31, .391, .365, .185, None],
}
STATS_REACTION_DIFFUSION = {
  'mean': [0.],
  'std': [1.],
}
STATS_WAVE_EQUATION = {
  'mean': [0.],
  'std': [1.],
}

DATASET_METADATA = {
  # incompressible_fluids: [velocity, velocity]
  'incompressible_fluids/brownian_bridge': Metadata(
    data_group='velocity',
    active_variables=ACTIVE_VARS_INCOMPRESSIBLE_FLUIDS,
    target_variables=TARGET_VARS_INCOMPRESSIBLE_FLUIDS,
    stats=STATS_INCOMPRESSIBLE_FLUIDS,
    signed=True,
  ),
  'incompressible_fluids/gaussians': Metadata(
    data_group='velocity',
    active_variables=ACTIVE_VARS_INCOMPRESSIBLE_FLUIDS,
    target_variables=TARGET_VARS_INCOMPRESSIBLE_FLUIDS,
    stats=STATS_INCOMPRESSIBLE_FLUIDS,
    signed=True,
  ),
  'incompressible_fluids/pwc': Metadata(
    data_group='velocity',
    active_variables=ACTIVE_VARS_INCOMPRESSIBLE_FLUIDS,
    target_variables=TARGET_VARS_INCOMPRESSIBLE_FLUIDS,
    stats=STATS_INCOMPRESSIBLE_FLUIDS,
    signed=True,
  ),
  'incompressible_fluids/shear_layer': Metadata(
    data_group='velocity',
    active_variables=ACTIVE_VARS_INCOMPRESSIBLE_FLUIDS,
    target_variables=TARGET_VARS_INCOMPRESSIBLE_FLUIDS,
    stats=STATS_INCOMPRESSIBLE_FLUIDS,
    signed=True,
  ),
  'incompressible_fluids/sines': Metadata(
    data_group='velocity',
    active_variables=ACTIVE_VARS_INCOMPRESSIBLE_FLUIDS,
    target_variables=TARGET_VARS_INCOMPRESSIBLE_FLUIDS,
    stats=STATS_INCOMPRESSIBLE_FLUIDS,
    signed=True,
  ),
  'incompressible_fluids/vortex_sheet': Metadata(
    data_group='velocity',
    active_variables=ACTIVE_VARS_INCOMPRESSIBLE_FLUIDS,
    target_variables=TARGET_VARS_INCOMPRESSIBLE_FLUIDS,
    stats=STATS_INCOMPRESSIBLE_FLUIDS,
    signed=True,
  ),
  # compressible_flow: [density, velocity, velocity, pressure, energy]
  'compressible_flow/cloudshock': Metadata(
    data_group='data',
    active_variables=ACTIVE_VARS_COMPRESSIBLE_FLOW,
    target_variables=TARGET_VARS_COMPRESSIBLE_FLOW,
    stats=STATS_COMPRESSIBLE_FLOW,
    signed=[False, True, True, False, False],
  ),
  'compressible_flow/gauss': Metadata(
    data_group='data',
    active_variables=ACTIVE_VARS_COMPRESSIBLE_FLOW,
    target_variables=TARGET_VARS_COMPRESSIBLE_FLOW,
    stats=STATS_COMPRESSIBLE_FLOW,
    signed=[False, True, True, False, False],
  ),
  'compressible_flow/kh': Metadata(
    data_group='data',
    active_variables=ACTIVE_VARS_COMPRESSIBLE_FLOW,
    target_variables=TARGET_VARS_COMPRESSIBLE_FLOW,
    stats=STATS_COMPRESSIBLE_FLOW,
    signed=[False, True, True, False, False],
  ),
  'compressible_flow/richtmyer_meshkov': Metadata(
    data_group='solution',
    active_variables=ACTIVE_VARS_COMPRESSIBLE_FLOW,
    target_variables=TARGET_VARS_COMPRESSIBLE_FLOW,
    stats=STATS_COMPRESSIBLE_FLOW,
    signed=[False, True, True, False, False],
  ),
  'compressible_flow/riemann': Metadata(
    data_group='data',
    active_variables=ACTIVE_VARS_COMPRESSIBLE_FLOW,
    target_variables=TARGET_VARS_COMPRESSIBLE_FLOW,
    stats=STATS_COMPRESSIBLE_FLOW,
    signed=[False, True, True, False, False],
  ),
  'compressible_flow/riemann_curved': Metadata(
    data_group='data',
    active_variables=ACTIVE_VARS_COMPRESSIBLE_FLOW,
    target_variables=TARGET_VARS_COMPRESSIBLE_FLOW,
    stats=STATS_COMPRESSIBLE_FLOW,
    signed=[False, True, True, False, False],
  ),
  'compressible_flow/riemann_kh': Metadata(
    data_group='data',
    active_variables=ACTIVE_VARS_COMPRESSIBLE_FLOW,
    target_variables=TARGET_VARS_COMPRESSIBLE_FLOW,
    stats=STATS_COMPRESSIBLE_FLOW,
    signed=[False, True, True, False, False],
  ),
  'compressible_flow/gravity/blast': Metadata(
    data_group='solution',
    active_variables=ACTIVE_VARS_COMPRESSIBLE_FLOW,
    target_variables=TARGET_VARS_COMPRESSIBLE_FLOW,
    stats=STATS_COMPRESSIBLE_FLOW,
    signed=[False, True, True, False, False],
  ),
  'compressible_flow/gravity/rayleigh_taylor': Metadata(
    data_group='solution',
    active_variables=ACTIVE_VARS_COMPRESSIBLE_FLOW,
    target_variables=TARGET_VARS_COMPRESSIBLE_FLOW,
    stats=STATS_COMPRESSIBLE_FLOW,
    signed=[False, True, True, False, False],
  ),
  # reaction_diffusion
  'reaction_diffusion/allen_cahn': Metadata(
    data_group='solution',
    active_variables=[0],
    target_variables=[0],
    stats=STATS_REACTION_DIFFUSION,
    signed=True,
  ),
  # wave_equation
  'wave_equation/seismic_20step': Metadata(
    data_group='solution',
    active_variables=[0],
    target_variables=[0],
    stats=STATS_WAVE_EQUATION,
    signed=True,
  ),
}

class Dataset:

  def __init__(self,
    key: flax.typing.PRNGKey,
    datadir: str,
    datapath: str,
    n_train: int = 0,
    n_valid: int = 0,
    preload: bool = False,
    include_passive_variables: bool = False,
    downsample_factor: int = 1,
    cutoff: int = None,
  ):

    # Set attributes
    self.metadata = DATASET_METADATA[datapath]
    self.data_group = self.metadata.data_group
    self.reader = h5py.File(Path(datadir) / f'{datapath}.nc', 'r')
    self.idx_vars = (None if include_passive_variables
      else self.metadata.active_variables)
    self.preload = preload
    self.data = None
    self.length = ((n_train + n_valid) if self.preload
      else self.reader[self.data_group].shape[0])
    self.cutoff = cutoff if (cutoff is not None) else (self._fetch(0, raw=True)[0].shape[1])
    self.downsample_factor = downsample_factor
    self.sample = self._fetch(0)
    self.shape = self.sample[0].shape

    # Split the dataset
    assert (n_train + n_valid) <= self.length
    self.nums = {'train': n_train, 'valid': n_valid}
    self.idx_modes = {
      'train': jax.random.permutation(key, n_train),  # First n_train samples
      'valid': np.arange((self.length - n_valid), self.length),  # Last n_valid samples
    }

    # Instantiate the dataset stats
    self.stats = {
      'trj': {'mean': None, 'std': None},
      'der': {'mean': None, 'std': None},
      'res': {'mean': None, 'std': None},
      'time': {'max': self.shape[1]},
    }

    if self.preload:
      _len_dataset = self.reader[self.data_group].shape[0]
      train_data = self.reader[self.data_group][np.arange(n_train)]
      valid_data = self.reader[self.data_group][np.arange(_len_dataset - n_valid, _len_dataset)]
      self.data = np.concatenate([train_data, valid_data], axis=0)

  def compute_stats(self,
      axes: Sequence[int] = (0,),
      derivs_degree: int = 0,
      residual_steps: int = 0,
      skip_residual_steps: int = 1,
    ) -> None:

    # Check inputs
    assert residual_steps >= 0
    assert residual_steps < self.shape[1]
    assert 1 in axes  # NOTE: Otherwise we cannot extrapolate in time

    # Get all trajectories
    trj, _ = self.train(np.arange(self.nums['train']))

    # Compute statistics of the solutions
    self.stats['trj']['mean'] = np.mean(trj, axis=axes, keepdims=True)
    self.stats['trj']['std'] = np.std(trj, axis=axes, keepdims=True)

    # Compute statistics of the derivatives
    if derivs_degree > 0:
      trj_nrm = normalize(trj, shift=self.stats['trj']['mean'], scale=self.stats['trj']['std'])
      trj_nrm_der = compute_derivatives(trj_nrm, degree=derivs_degree)
      self.stats['der']['mean'] = np.mean(trj_nrm_der, axis=axes, keepdims=True)
      self.stats['der']['std'] = np.std(trj_nrm_der, axis=axes, keepdims=True)

    # Compute statistics of the residuals
    # TRY: Compute statistics of residuals of normalized trajectories
    _get_res = lambda s, trj: trj[:, (s):] - trj[:, :-(s)]
    residuals = []
    for s in range(1, residual_steps+1):
      if (s % skip_residual_steps):
        continue
      res = _get_res(s, trj)
      residuals.append(res)
    residuals = np.concatenate(residuals, axis=1)

    self.stats['res']['mean'] = np.mean(residuals, axis=axes, keepdims=True)
    self.stats['res']['std'] = np.std(residuals, axis=axes, keepdims=True)

  def _fetch(self, idx: Union[int, Sequence], raw: bool = False):
    """Fetches a sample from the dataset, given its global index."""

    # Check inputs
    if isinstance(idx, int):
      idx = [idx]

    # Get trajectories
    if self.data is not None:
      traj = self.data[np.sort(idx)]
    else:
      traj = self.reader[self.data_group][np.sort(idx)]
    # Move axes
    if len(traj.shape) == 5:
      traj = np.moveaxis(traj, source=(2, 3, 4), destination=(4, 2, 3))
    elif len(traj.shape) == 4:
      traj = np.expand_dims(traj, axis=-1)
    # Set equation parameters
    spec = None

    # Select variables
    if self.idx_vars is not None:
      traj = traj[..., self.idx_vars]

    # Downsample and cut the trajectories
    if not raw:
      traj = traj[:, ::self.downsample_factor]
      traj = traj[:, :self.cutoff]

    return traj, spec

  def _fetch_mode(self, idx: Union[int, Sequence], mode: str):
    # Check inputs
    if isinstance(idx, int):
      idx = [idx]
    # Set mode index
    assert all([i < len(self.idx_modes[mode]) for i in idx])
    _idx = self.idx_modes[mode][np.array(idx)]

    return self._fetch(_idx)

  def train(self, idx: Union[int, Sequence]):
    return self._fetch_mode(idx, mode='train')

  def valid(self, idx: Union[int, Sequence]):
    return self._fetch_mode(idx, mode='valid')

  def batches(self, mode: str, batch_size: int, key: flax.typing.PRNGKey = None):
    assert batch_size > 0
    assert batch_size <= self.nums[mode]

    if key is not None:
      _idx_mode_permuted = jax.random.permutation(key, np.arange(self.nums[mode]))
    else:
      _idx_mode_permuted = jnp.arange(self.nums[mode])

    len_dividable = self.nums[mode] - (self.nums[mode] % batch_size)
    for idx in np.split(_idx_mode_permuted[:len_dividable], len_dividable // batch_size):
      batch = self._fetch_mode(idx, mode)
      yield batch

    if (self.nums[mode] % batch_size):
      idx = _idx_mode_permuted[len_dividable:]
      batch = self._fetch_mode(idx, mode)
      yield batch

  def __len__(self):
    return self.length
