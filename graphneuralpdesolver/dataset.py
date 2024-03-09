"""Utility functions for reading the datasets."""

from pathlib import Path
import h5py
import numpy as np
from typing import Any, Union, Sequence, Tuple

import numpy as np
import jax
import jax.lax
import flax.typing

from graphneuralpdesolver.utils import Array


NX_SUPER_RESOLUTION = 256
NT_SUPER_RESOLUTION = 256

class Dataset:

  def __init__(self, dir: str, n_train: int, n_valid: int, n_test: int):
    self.reader = h5py.File(dir, 'r')
    self.length = len([k for k in self.reader.keys() if 'sample' in k])

    # Split the dataset
    assert (n_train+n_valid+n_test) < self.length
    self.nums = {'train': n_train, 'valid': n_valid, 'test': n_test}
    random_permutation = np.random.permutation(self.length)  # TODO: Do it with jax for reproducibility
    self.idx_modes = {
      'train': random_permutation[:n_train],
      'valid': random_permutation[n_train:(n_train+n_valid)],
      'test': random_permutation[(n_train+n_valid):(n_train+n_valid+n_test)],
    }

    # Compute mean
    _sum = np.zeros_like(self._fetch(0))
    for idx in range(n_train):
      _sum += self.train(idx)
    self.mean = _sum / n_train

    # Compute std
    _sum = np.zeros_like(self._fetch(0))
    for idx in range(n_train):
      _sum += np.power(self.train(idx) - self.mean, 2)
    self.std_train = np.sqrt(_sum / n_train)

  def _fetch(self, idx):
    traj = self.reader[f'sample_{str(idx)}'][:]
    traj = traj[None, ...]
    traj = np.moveaxis(traj, source=(2, 3, 4), destination=(4, 2, 3))

    return traj

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

  def batches(self, mode: str, batch_size: int):
    assert batch_size > 0
    assert batch_size <= self.nums[mode]
    _idx_next = 0
    for _ in range(self.nums[mode] // batch_size):
      batch = np.concatenate(
        [self._fetch_mode(_idx, mode) for _idx in range(_idx_next, _idx_next+batch_size)],
      )
      _idx_next += batch_size
      yield batch
    rem = (self.nums[mode] % batch_size)
    if rem:
      batch = np.concatenate(
        [self._fetch_mode(_idx, mode) for _idx in range(_idx_next, _idx_next+rem)],
      )
      yield batch

  def __len__(self):
    return self.length

def read_datasets(dir: Union[Path, str], pde_type: str, experiment: str, nx: int,
                  downsample_x: bool = True,
                  modes: Sequence[str] = ['train', 'valid', 'test'],
                ) -> dict[str, dict[str, Any]]:
  """Reads a dataset from its source file and prepares the shapes and specifications."""

  dir = Path(dir)
  assert dir.is_dir(), f'The path {dir} is not a directory.'
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

def _read_dataset_attributes(h5group: h5py.Group, nx: int, nt: int) -> dict[str, Any]:
  """Prepares the shapes and puts together the specifications of a dataset."""

  resolution = f'pde_{nt}-{nx}'
  dataset = dict(
    # TODO: Different for different pdes
    specs = np.stack([h5group['alpha'][:], h5group['beta'][:], h5group['gamma'][:]], axis=-1),
    trajectories = h5group[resolution][:][..., None],
    x = h5group[resolution].attrs['x'],
    # dx = h5group[resolution].attrs['dx'].item(),
    tmin = h5group[resolution].attrs['tmin'].item(),
    tmax = h5group[resolution].attrs['tmax'].item(),
    dt = h5group[resolution].attrs['dt'].item(),
    # nx = h5group[resolution].attrs['nx'].item(),
    # nt = h5group[resolution].attrs['nt'].item(),
  )
  dataset['dx'] = (dataset['x'][1] - dataset['x'][0]).item()  # CHECK: Why is it different from data['dx']?
  dataset['range_x'] = (0., 16.)

  return dataset

def shuffle_arrays(key: flax.typing.PRNGKey, arrays: Sequence[Array]) -> Sequence[Array]:
  """Shuffles a set of arrays with the same random permutation."""

  size = arrays[0].shape[0]
  assert all([arr.shape[0] == size for arr in arrays])
  permutation = jax.random.permutation(key, size)

  return [arr[permutation] for arr in arrays]

def normalize(trajectories: Array, stats: Tuple[Array, Array] = None):
  if stats:
    mean, std = stats
  else:
    mean = np.mean(trajectories, axis=0)[None, :]
    std = np.std(trajectories, axis=0)[None, :]

  trajectories = (trajectories - mean) / std

  return trajectories, (mean, std)

def unnormalize(trajectories: Array, stats: Tuple[Array, Array]):
  mean, std = stats
  trajectories = std * trajectories + mean

  return trajectories

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

def downsample(arr: Array, ratio: int, axis: int = 0) -> Array:
  slc = [slice(None)] * len(arr.shape)
  slc[axis] = slice(None, None, ratio)
  return arr[tuple(slc)]
