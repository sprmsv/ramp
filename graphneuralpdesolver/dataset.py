"""Utility functions for reading the datasets."""

from pathlib import Path
import h5py
import numpy as np
from typing import Any, Union, Sequence

from absl import logging
import numpy as np
import jax.numpy as jnp
import jax

Array = Union[jnp.ndarray, np.ndarray]
KeyArray = Array

def read_datasets(dir: Union[Path, str], pde_type: str, experiment: str, nx: int,
                  modes: Sequence[str] = ['train', 'valid', 'test'],
                ) -> dict[str, dict[str, Any]]:
  """Reads a dataset from its source file and prepares the shapes and specifications."""

  dir = Path(dir)
  assert dir.is_dir(), f"The path {dir} is not a directory."
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

def prepare_dataset(key: KeyArray, dataset: dict[str, Any], batch_size: int,
                    num_times_input: int, num_times_output: int,
                    shuffle_batches: bool = False, shuffle_leadtime: bool = False
                    ) -> dict[str, Array]:
  """
  Creates batches of trajectories and splits each batch into
  chunks of inputs and labels with different lead times.
  """

  trajectories, specs = dataset['trajectories'], dataset['specs']

  # Shuffle both arrays before creating batches
  if shuffle_batches:
    subkey, key = jax.random.split(key)
    trajectories, specs = _shuffle_arrays(subkey, [trajectories, specs])

  # Split in batches
  n_batches = jnp.ceil(trajectories.shape[0] / batch_size)
  trajectories = jnp.split(trajectories, jnp.arange(1, n_batches) * batch_size)
  specs = jnp.split(specs, jnp.arange(1, n_batches) * batch_size)

  # Create inputs and labels with different lead times for each batch.
  batches = []
  for trajectories_batch, specs_batch in zip(trajectories, specs):
    batch = _split_trajectories(trajectories_batch, specs_batch, K_inp=num_times_input, K_out=num_times_output)
    if shuffle_leadtime:
      subkey, key = jax.random.split(key)
      batch['inputs'], batch['specs'], batch['labels'] = _shuffle_arrays(
        subkey, [batch['inputs'], batch['specs'], batch['labels']])
    batches.append(batch)

  return batches

def _shuffle_arrays(key: KeyArray, arrays: Sequence[Array]) -> Sequence[Array]:
  """Shuffles a set of arrays with the same random permutation."""

  size = arrays[0].shape[0]
  assert all([arr.shape[0] == size for arr in arrays])
  permutation = jax.random.permutation(key, size)

  return [arr[permutation] for arr in arrays]

def _split_trajectories(trajectories: Array, specs: Array, K_inp: int, K_out: int) -> dict[jnp.ndarray]:

    assert trajectories.shape[0] == specs.shape[0]
    nt = trajectories.shape[1]
    permissible_starting_points = jnp.arange(start=K_inp-1, stop=nt-K_out)
    starting_points = permissible_starting_points

    data = {
        'inputs': jnp.concatenate([trajectories[:, s-K_inp+1:s+1, :, :] for s in starting_points], axis=0),
        'labels': jnp.concatenate([trajectories[:, s+1:s+K_out+1, :, :] for s in starting_points], axis=0),
        'specs': jnp.concatenate([specs] * starting_points.shape[0], axis=0),
    }

    return data
