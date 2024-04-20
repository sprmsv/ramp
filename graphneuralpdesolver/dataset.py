"""Utility functions for reading the datasets."""

from pathlib import Path
import h5py
import numpy as np
from typing import Any, Union, Sequence, Tuple

import numpy as np
import jax
import jax.lax
import jax.numpy as jnp
import flax.typing

from graphneuralpdesolver.utils import Array


NX_SUPER_RESOLUTION = 256
NT_SUPER_RESOLUTION = 256

DATAGROUP = {
  'incompressible_fluids': 'velocity',
  'compressible_flow': 'data',
}

# TODO: Remove
class Dataset_old:

  def __init__(self, key: flax.typing.PRNGKey, dir: str,
      n_train: int, n_valid: int, n_test: int,
      cutoff: int = None, downsample_factor: int = 1,
    ):
    self.reader = h5py.File(dir, 'r')
    self.length = len([k for k in self.reader.keys() if 'sample' in k])
    self.cutoff = cutoff if (cutoff is not None) else (self._fetch(0, raw=True)[0].shape[1])
    self.downsample_factor = downsample_factor
    self.sample = self._fetch(0)
    self.shape = self.sample[0].shape

    # Split the dataset
    assert (n_train+n_valid+n_test) <= self.length
    self.nums = {'train': n_train, 'valid': n_valid, 'test': n_test}
    random_permutation = jax.random.permutation(key, self.length)
    self.idx_modes = {
      'train': random_permutation[:n_train],
      'valid': random_permutation[n_train:(n_train+n_valid)],
      'test': random_permutation[(n_train+n_valid):(n_train+n_valid+n_test)],
    }

    # Instantiate the dataset stats
    self.mean_trj, self.std_trj = None, None
    self.mean_res, self.std_res = None, None

  def compute_stats(self, residual_steps: int = 0, skip_residual_steps: int = 1):
    assert residual_steps >= 0
    assert residual_steps < self.shape[1]

    # Compute mean of trajectories
    if self.mean_trj is None:
      _mean_samples = np.zeros_like(self.sample[0])
      for idx in range(self.nums['train']):
        _mean_samples += self.train(idx)[0] / self.nums['train']
      self.mean_trj = np.mean(_mean_samples, axis=1, keepdims=True)

    # Compute std of trajectories
    if self.std_trj is None:
      _mean_samples = np.zeros_like(self.sample[0])
      for idx in range(self.nums['train']):
        _mean_samples += np.power(self.train(idx)[0] - self.mean_trj, 2) / self.nums['train']
      self.std_trj = np.sqrt(np.mean(_mean_samples, axis=1, keepdims=True))

    if residual_steps > 0:
      _get_res = lambda s, trj: trj[:, (s):] - trj[:, :-(s)]

      # Compute mean of residuals
      _mean_samples = [
        np.zeros_like(_get_res(s, self.sample[0]))
        for s in range(1, residual_steps+1)
      ]
      for idx in range(self.nums['train']):
        trj = self.train(idx)[0]
        for s in range(1, residual_steps+1):
          if (s % skip_residual_steps):
            continue
          _mean_samples[s-1] += _get_res(s, trj) / self.nums['train']
      self.mean_res = [
        np.mean(_mean_samples[s-1], axis=1, keepdims=True)
        for s in range(1, residual_steps+1)
      ]

      # Compute std of residuals
      _mean_samples = [
        np.zeros_like(_get_res(s, self.sample[0]))
        for s in range(1, residual_steps+1)
      ]
      for idx in range(self.nums['train']):
        trj = self.train(idx)[0]
        for s in range(1, residual_steps+1):
          if (s % skip_residual_steps):
            continue
          _mean_samples[s-1] += np.power(
            _get_res(s, trj) - self.mean_res[s-1], 2) / self.nums['train']
      self.std_res = [
        np.sqrt(np.mean(_mean_samples[s-1], axis=1, keepdims=True))
        for s in range(1, residual_steps+1)
      ]

  def _fetch(self, idx: int, raw: bool = False):
    traj = self.reader[f'sample_{str(idx)}'][:]
    traj = traj[None, ...]
    traj = np.moveaxis(traj, source=(2, 3, 4), destination=(4, 2, 3))
    spec = None

    if not raw:
      traj = traj[:, :(self.cutoff):self.downsample_factor]

    return traj, spec

  def _fetch_mode(self, idx, mode):
    assert idx < len(self.idx_modes[mode])
    _idx = self.idx_modes[mode][idx]
    return self._fetch(_idx)

  def train(self, idx):
    return self._fetch_mode(idx, 'train')

  def valid(self, idx):
    return self._fetch_mode(idx, 'valid')

  def test(self, idx):
    return self._fetch_mode(idx, 'test')

  def batches(self, mode: str, batch_size: int, key: flax.typing.PRNGKey = None):
    assert batch_size > 0
    assert batch_size <= self.nums[mode]
    _idx_next = 0
    _idx_mode_permuted = jnp.arange(self.nums[mode])
    if key is not None:
      _idx_mode_permuted = jax.random.permutation(key, np.arange(self.nums[mode]))
    for _ in range(self.nums[mode] // batch_size):
      trajs_batch = np.concatenate([
        self._fetch_mode(_idx_mode_permuted[_idx], mode)[0]
        for _idx in range(_idx_next, _idx_next+batch_size)
      ])
      specs_batch = None
      _idx_next += batch_size
      yield trajs_batch, specs_batch
    rem = (self.nums[mode] % batch_size)
    if rem:
      trajs_batch = np.concatenate([
        self._fetch_mode(_idx_mode_permuted[_idx], mode)[0]
        for _idx in range(_idx_next, _idx_next+rem)
      ])
      specs_batch = None
      yield trajs_batch, specs_batch

  def __len__(self):
    return self.length

class Dataset:

  def __init__(self, key: flax.typing.PRNGKey,
      datadir: str, subpath: str, name: str,
      n_train: int, n_valid: int, n_test: int,
      idx_vars: Union[int, Sequence] = None,
      cutoff: int = None, downsample_factor: int = 1,
    ):
    self.datagroup = DATAGROUP[subpath]
    self.reader = h5py.File(Path(datadir) / subpath / f'{name}.nc', 'r')
    self.length = self.reader[self.datagroup].shape[0]
    self.idx_vars = [idx_vars] if isinstance(idx_vars, int) else idx_vars
    self.cutoff = cutoff if (cutoff is not None) else (self._fetch(0, raw=True)[0].shape[1])
    self.downsample_factor = downsample_factor
    self.sample = self._fetch(0)
    self.shape = self.sample[0].shape

    # Split the dataset
    assert (n_train+n_valid+n_test) <= self.length
    self.nums = {'train': n_train, 'valid': n_valid, 'test': n_test}
    random_permutation = jax.random.permutation(key, self.length)
    self.idx_modes = {
      'train': random_permutation[:n_train],
      'valid': random_permutation[n_train:(n_train+n_valid)],
      'test': random_permutation[(n_train+n_valid):(n_train+n_valid+n_test)],
    }

    # Instantiate the dataset stats
    self.stats = {
      'trj': {'mean': None, 'std': None},
      'res': {'mean': None, 'std': None},
    }

  def compute_stats(self, residual_steps: int = 0,
      skip_residual_steps: int = 1, batch_size: int = None) -> None:
    assert residual_steps >= 0
    assert residual_steps < self.shape[1]

    _get_res = lambda s, trj: trj[:, (s):] - trj[:, :-(s)]

    if batch_size is None:
      # Get all trajectories
      trj, _ = self.train(np.arange(self.nums['train']))

      # Compute statistics of the solutions
      self.stats['trj']['mean'] = np.mean(trj, axis=(0, 1), keepdims=True)
      self.stats['trj']['std'] = np.std(trj, axis=(0, 1), keepdims=True)

      # Compute statistics of the residuals
      self.stats['res']['mean'] = []
      self.stats['res']['std'] = []
      for s in range(1, residual_steps+1):
        if (s % skip_residual_steps):
          self.stats['res']['mean'].append(np.zeros(shape=(1, 1, *self.shape[2:])))
          self.stats['res']['std'].append(np.zeros(shape=(1, 1, *self.shape[2:])))
        res = _get_res(s, trj)
        self.stats['res']['mean'].append(np.mean(res, axis=(0, 1), keepdims=True))
        self.stats['res']['std'].append(np.std(res, axis=(0, 1), keepdims=True))

    else:
      # Compute mean of trajectories
      if self.stats['trj']['mean'] is None:
        _mean_samples = np.zeros_like(self.sample[0])
        for trj, _ in self.batches(mode='train', batch_size=batch_size):
          _mean_samples += np.sum(
            trj, axis=0, keepdims=True) / self.nums['train']
        self.stats['trj']['mean'] = np.mean(_mean_samples, axis=1, keepdims=True)

      # Compute std of trajectories
      if self.stats['trj']['std'] is None:
        _mean_samples = np.zeros_like(self.sample[0])
        for trj, _ in self.batches(mode='train', batch_size=batch_size):
          _mean_samples += np.sum(np.power(
            trj - self.stats['trj']['mean'], 2), axis=0, keepdims=True
          ) / self.nums['train']
        self.stats['trj']['std'] = np.sqrt(np.mean(_mean_samples, axis=1, keepdims=True))

      if residual_steps > 0:

        # Compute mean of residuals
        _mean_samples = [
          np.zeros_like(_get_res(s, self.sample[0]))
          for s in range(1, residual_steps+1)
        ]
        for trj, _ in self.batches(mode='train', batch_size=batch_size):
          for s in range(1, residual_steps+1):
            if (s % skip_residual_steps):
              continue
            _mean_samples[s-1] += np.sum(
              _get_res(s, trj), axis=0, keepdims=True) / self.nums['train']
        self.stats['res']['mean'] = [
          np.mean(_mean_samples[s-1], axis=1, keepdims=True)
          for s in range(1, residual_steps+1)
        ]

        # Compute std of residuals
        _mean_samples = [
          np.zeros_like(_get_res(s, self.sample[0]))
          for s in range(1, residual_steps+1)
        ]
        for trj, _ in self.batches(mode='train', batch_size=batch_size):
          for s in range(1, residual_steps+1):
            if (s % skip_residual_steps):
              continue
            _mean_samples[s-1] += np.sum(np.power(
              _get_res(s, trj) - self.stats['res']['mean'][s-1], 2), axis=0, keepdims=True
            ) / self.nums['train']
        self.stats['res']['std'] = [
          np.sqrt(np.mean(_mean_samples[s-1], axis=1, keepdims=True))
          for s in range(1, residual_steps+1)
        ]

  def _fetch(self, idx: Union[int, Sequence], raw: bool = False):
    if isinstance(idx, int):
      idx = [idx]
    traj = self.reader[self.datagroup][np.sort(idx)]
    traj = np.moveaxis(traj, source=(2, 3, 4), destination=(4, 2, 3))
    spec = None

    if self.idx_vars is not None:
      traj = traj[..., self.idx_vars]

    if not raw:
      traj = traj[:, :(self.cutoff):self.downsample_factor]

    return traj, spec

  def _fetch_mode(self, idx: Union[int, Sequence], mode: str):
    if isinstance(idx, int):
      idx = [idx]
    assert all([i < len(self.idx_modes[mode]) for i in idx])
    _idx = self.idx_modes[mode][np.array(idx)]
    return self._fetch(_idx)

  def train(self, idx: Union[int, Sequence]):
    return self._fetch_mode(idx, 'train')

  def valid(self, idx: Union[int, Sequence]):
    return self._fetch_mode(idx, 'valid')

  def test(self, idx: Union[int, Sequence]):
    return self._fetch_mode(idx, 'test')

  def batches(self, mode: str, batch_size: int, key: flax.typing.PRNGKey = None):
    assert batch_size > 0
    assert batch_size <= self.nums[mode]

    if key is not None:
      _idx_mode_permuted = jax.random.permutation(key, np.arange(self.nums[mode]))
    else:
      _idx_mode_permuted = jnp.arange(self.nums[mode])

    len_dividable = self.nums[mode] - (self.nums[mode] % batch_size)
    for idx in np.split(_idx_mode_permuted[:len_dividable], len_dividable // batch_size):
      trajs_batch = self._fetch_mode(idx, mode)[0]
      specs_batch = None
      yield trajs_batch, specs_batch

    if (self.nums[mode] % batch_size):
      idx = _idx_mode_permuted[len_dividable:]
      trajs_batch = self._fetch_mode(idx, mode)[0]
      specs_batch = None
      yield trajs_batch, specs_batch

  def __len__(self):
    return self.length

# NOTE: 1D
def read_datasets(dir: Union[Path, str], pde_type: str, experiment: str, nx: int,
                  downsample_x: bool = True,
                  modes: Sequence[str] = ['train', 'valid', 'test'],
                ) -> dict[str, dict[str, Any]]:
  """Reads a dataset from its source file and prepares the shapes and specifications."""

  dir = Path(dir)
  datasets = {
    mode: _read_dataset_attributes(
      h5group=h5py.File(dir / f'{pde_type}_{mode}_{experiment}.h5')[mode],
      nx=(NX_SUPER_RESOLUTION if downsample_x else nx),
      nt=NT_SUPER_RESOLUTION,
    )
    for mode in modes
  }

  if downsample_x:
    ratio = NX_SUPER_RESOLUTION // nx
    for dataset in datasets.values():
      dataset['trajectories'] = downsample(dataset['trajectories'], ratio=ratio, axis=2)
      dataset['x'] = downsample(dataset['x'], ratio=ratio, axis=0)
      dataset['dx'] = dataset['dx'] * ratio

  return datasets

# NOTE: 1D
def _read_dataset_attributes(h5group: h5py.Group, nx: int, nt: int) -> dict[str, Any]:
  """Prepares the shapes and puts together the specifications of a dataset."""

  resolution = f'pde_{nt}-{nx}'
  trajectories = h5group[resolution][:]
  if trajectories.ndim == 3:
    trajectories = trajectories[:, None]
  dataset = dict(
    trajectories = np.moveaxis(trajectories, (1, 2, 3), (3, 1, 2)),
    x = h5group[resolution].attrs['x'],
    # dx = h5group[resolution].attrs['dx'].item(),
    tmin = h5group[resolution].attrs['tmin'].item(),
    tmax = h5group[resolution].attrs['tmax'].item(),
    dt = h5group[resolution].attrs['dt'].item(),
    # nx = h5group[resolution].attrs['nx'].item(),
    # nt = h5group[resolution].attrs['nt'].item(),
  )
  dataset['dx'] = (dataset['x'][1] - dataset['x'][0]).item()  # CHECK: Why is it different from data['dx']?
  dataset['range_x'] = (np.min(dataset['x']), np.max(dataset['x']))

  return dataset

def shuffle_arrays(key: flax.typing.PRNGKey, arrays: Sequence[Array]) -> Sequence[Array]:
  """Shuffles a set of arrays with the same random permutation along the first axis."""

  size = arrays[0].shape[0]
  assert all([arr.shape[0] == size for arr in arrays])
  permutation = jax.random.permutation(key, size)

  return [arr[permutation] for arr in arrays]

def normalize(arr: Array, mean: Array, std: Array):
  std = jnp.where(std == 0., 1., std)
  arr = (arr - mean) / std
  return arr

def unnormalize(arr: Array, mean: Array, std: Array):
  arr = std * arr + mean
  return arr

# NOTE: 1D
def downsample_convolution(trajectories: Array, ratio: int) -> Array:
  trj_padded = np.concatenate([trajectories[:, :, -(ratio//2+1):-1], trajectories, trajectories[:, :, :(ratio//2)]], axis=2)
  kernel = np.array([1/(ratio+1)]*(ratio+1)).reshape(1, 1, ratio+1, 1)
  trj_downsampled = jax.lax.conv_general_dilated(
    lhs=trj_padded,
    rhs=kernel,
    window_strides=(1,ratio),
    padding='VALID',
    dimension_numbers=jax.lax.conv_dimension_numbers(
      trajectories.shape, kernel.shape, ('NHWC', 'OHWI', 'NHWC')),
  )

  return trj_downsampled

# NOTE: 1D
def downsample(arr: Array, ratio: int, axis: int = 0) -> Array:
  slc = [slice(None)] * len(arr.shape)
  slc[axis] = slice(None, None, ratio)
  return arr[tuple(slc)]
