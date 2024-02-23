"""Utility functions for reading the datasets."""

from pathlib import Path
import h5py
import numpy as np
from typing import Any, Union, Sequence

import numpy as np
import jax
import flax.typing

from graphneuralpdesolver.utils import Array


def read_datasets(dir: Union[Path, str], pde_type: str, experiment: str, nx: int,
                  modes: Sequence[str] = ['train', 'valid', 'test'],
                ) -> dict[str, dict[str, Any]]:
  """Reads a dataset from its source file and prepares the shapes and specifications."""

  dir = Path(dir)
  assert dir.is_dir(), f'The path {dir} is not a directory.'
  datasets = {
    mode: _read_dataset_attributes(
      h5group=h5py.File(dir / f'{pde_type}_{mode}_{experiment}.h5')[mode],
      nx=nx, nt=256,
    )
    for mode in modes
  }

  return datasets

def _read_dataset_attributes(h5group: h5py.Group, nx: int, nt: int) -> dict[str, Any]:
  """Prepares the shapes and puts together the specifications of a dataset."""

  resolution = f'pde_{nt}-{nx}'
  dataset = dict(
    # TODO: Different for different pdes
    specs = np.stack([h5group['alpha'][:], h5group['beta'][:], h5group['gamma'][:]], axis=-1),
    trajectories = h5group[resolution][:][..., None],
    x = h5group[resolution].attrs['x'],
    tmin = h5group[resolution].attrs['tmin'].item(),
    tmax = h5group[resolution].attrs['tmax'].item(),
    # dx = h5group[resolution].attrs['dx'].item(),
    # dt = h5group[resolution].attrs['dt'].item(),
    # nx = h5group[resolution].attrs['nx'].item(),
    # nt = h5group[resolution].attrs['nt'].item(),
  )
  dataset['dx'] = (dataset['x'][1] - dataset['x'][0]).item()  # CHECK: Why is it different from data['dx']?
  dataset['domain_x'] = (0., 16.)

  return dataset

def shuffle_arrays(key: flax.typing.PRNGKey, arrays: Sequence[Array]) -> Sequence[Array]:
  """Shuffles a set of arrays with the same random permutation."""

  size = arrays[0].shape[0]
  assert all([arr.shape[0] == size for arr in arrays])
  permutation = jax.random.permutation(key, size)

  return [arr[permutation] for arr in arrays]
