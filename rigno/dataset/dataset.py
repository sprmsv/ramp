"""Classes for loading, processing, and manipulating the datasets."""

import h5py
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Sequence, NamedTuple, Mapping, Tuple
from copy import deepcopy

from flax.typing import PRNGKey
import jax
import jax.lax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np

from rigno.utils import Array
from rigno.dataset.metadata import DATASET_METADATA
from rigno.models.rigno import (
  RegionInteractionGraphMetadata,
  RegionInteractionGraphSet,
  RegionInteractionGraphBuilder)


@dataclass
class DiscretizedFunction:
  mask: Array
  values: Array

  def __repr__(self):
    return f'{self.__class__.__name__}({self.values.shape})'

class Batch(NamedTuple):
  x: Array
  t: Union[None, Array]
  g: Union[None, Sequence[RegionInteractionGraphSet]]
  functions: Mapping[str, DiscretizedFunction]

  @property
  def shape(self) -> tuple:
    return self.x.shape

  def __len__(self) -> int:
    return self.shape[0]

@dataclass
class Stats:
  mean: Array = None
  std: Array = None
  min: Array = None
  max: Array = None

class Dataset:

  def __init__(self,
    name: str,
    dir: str,
    file: str,
    time_cutoff_idx: int = None,
    time_downsample_factor: int = 1,
    space_downsample_factor: float = 1.,
    splits: Sequence[Tuple[int, int]] = None,
    preload: bool = False,
    key: PRNGKey = None,
  ):

    # Set attributes
    self.key = key if (key is not None) else jax.random.PRNGKey(0)
    self.metadata = deepcopy(DATASET_METADATA[name])
    self.time_cutoff_idx = time_cutoff_idx
    self.time_downsample_factor = time_downsample_factor
    self.space_downsample_factor = space_downsample_factor
    self.splits = splits
    if self.splits is None:
      self.splits = [(0, self.metadata.shape[0])]

    # Set data attributes
    self.rigs: RegionInteractionGraphMetadata = None
    file = h5py.File(Path(dir) / name / file, 'r')
    if preload:
      self.reader = flatten_nested_dictionaries(load_h5py_group_as_dictionary(file, splits=self.splits))
      len_splits = [(self.splits[i][1] - self.splits[i][0]) for i in range(len(self.splits))]
      split_borders = ([0] + np.cumsum(len_splits).tolist())
      self.splits = [(split_borders[i], split_borders[i+1]) for i in range(len(splits))]
    else:
      self.reader = file

    if self.time_dependent:
      raise NotImplementedError
      self.dt = ... # NOTE: Assuming fix dt

    # Instantiate the dataset stats
    self.stats = {key: Stats() for key in self.metadata.functions.keys()}
    self.stats_res = {key: Stats() for key in self.metadata.functions.keys()}
    self.stats_der = {key: Stats() for key in self.metadata.functions.keys()}
    self.stats['x'] = Stats(
      min=np.array(self.metadata.bbox_x[0]).reshape(1, 1, 1, -1),
      max=np.array(self.metadata.bbox_x[1]).reshape(1, 1, 1, -1),
    )
    self.stats['t'] = Stats(
      min=np.array(self.metadata.bbox_t[0]).reshape(1, 1, 1, 1),
      max=np.array(self.metadata.bbox_t[1]).reshape(1, 1, 1, 1),
    ) if self.time_dependent else Stats()

  @property
  def time_dependent(self):
    return self.metadata.bbox_t is not None

  def compute_stats(self, split: int = 0, residual_steps: int = 0) -> None:

    # Check inputs
    assert residual_steps >= 0
    assert residual_steps < self.metadata.shape[1]

    # Get all trajectories in a large batch
    # TODO: Avoid loading the whole dataset
    batch = self._get_batch(idx=np.arange(*self.splits[split]), get_graphs=False)

    # Compute statistics of functions
    for key in self.metadata.functions.keys():
      mask = batch.functions[key].mask
      values = batch.functions[key].values
      self.stats[key].mean = np.mean(values[:, :, np.where(mask)[2], :], axis=(0, 1, 2), keepdims=True)
      self.stats[key].std = np.std(values[:, :, np.where(mask)[2], :], axis=(0, 1, 2), keepdims=True)

    # Compute statistics of the residuals and time derivatives
    if self.time_dependent:
      _get_res = lambda s, trj: (trj[:, (s):] - trj[:, :-(s)])
      for key in self.metadata.functions.keys():
        mask = batch.functions[key].mask
        values = batch.functions[key].values
        residuals = []
        derivatives = []
        for s in range(1, residual_steps+1):
          res = _get_res(s, values)
          tau = _get_res(s, batch.t)
          residuals.append(res)
          derivatives.append(res / tau)
        residuals = np.concatenate(residuals, axis=1)
        derivatives = np.concatenate(derivatives, axis=1)
        self.stats_res[key].mean = np.mean(residuals[:, :, np.where(mask)[2], :], axis=(0, 1, 2), keepdims=True)
        self.stats_res[key].std = np.std(residuals[:, :, np.where(mask)[2], :], axis=(0, 1, 2), keepdims=True)
        self.stats_der[key].mean = np.mean(derivatives[:, :, np.where(mask)[2], :], axis=(0, 1, 2), keepdims=True)
        self.stats_der[key].std = np.std(derivatives[:, :, np.where(mask)[2], :], axis=(0, 1, 2), keepdims=True)

  def build_graphs(self, builder: RegionInteractionGraphBuilder, rmesh_correction_dsf: int = 1, key: PRNGKey = None) -> None:
    """Builds RIGNO graphs for all the samples in the dataset and stores them in the object."""
    # NOTE: Each graph takes about 3 MB and 2 seconds to build.
    # It can cause memory issues for large datasets.

    # NOTE: It is important to do the rmesh sub-sampling with a different key each time
    # Otherwise, for some datasets, the rmeshes can end up being similar
    if key is None:
      key = jax.random.PRNGKey(0)

    # Build graph metadata with potentially different number of edges
    # NOTE: Stores all graphs in memory one by one
    metadata = []
    num_p2r_edges = 0
    num_r2r_edges = 0
    num_r2p_edges = 0
    if self.rigs is not None:
      # NOTE: Use the old number of edges in order to avoid re-compilation
      num_p2r_edges = self.rigs.p2r_edge_indices.shape[1]
      num_r2r_edges = self.rigs.r2r_edge_indices.shape[1]
      if self.rigs.r2p_edge_indices is not None:
        num_r2p_edges = self.rigs.r2p_edge_indices.shape[1]
    for split in self.splits:
      if not (split[1] - split[0]) > 0: continue
      batch = self._get_batch(idx=np.arange(*split), get_graphs=False)
      # Loop over all coordinates in the batch
      # NOTE: Assuming constant x in time
      for x in batch.x[:, 0]:
        key, subkey = jax.random.split(key)
        m = builder.build_metadata(x_inp=x, x_out=x, domain=np.array(self.metadata.bbox_x), rmesh_correction_dsf=rmesh_correction_dsf, key=subkey)
        metadata.append(m)
        # Store the maximum number of edges
        if self.rigs is None:
          num_p2r_edges = max(num_p2r_edges, m.p2r_edge_indices.shape[1])
          num_r2r_edges = max(num_r2r_edges, m.r2r_edge_indices.shape[1])
          if m.r2p_edge_indices is not None:
            num_r2p_edges = max(num_r2p_edges, m.r2p_edge_indices.shape[1])
        # Break the loop if the coordinates are fixed on the batch axis
        if self.metadata.fix:
          break
      # Break the loop if the coordinates are fixed on the batch axis
      if self.metadata.fix:
        break

    # Pad the edge sets using dummy nodes
    # NOTE: Exploiting jax' behavior for out-of-dimension indexing
    for i, m in enumerate(metadata):
      m: RegionInteractionGraphMetadata
      metadata[i] = RegionInteractionGraphMetadata(
        x_pnodes_inp=m.x_pnodes_inp,
        x_pnodes_out=m.x_pnodes_out,
        x_rnodes=m.x_rnodes,
        r_rnodes=m.r_rnodes,
        p2r_edge_indices=m.p2r_edge_indices[:, jnp.arange(num_p2r_edges), :],
        r2r_edge_indices=m.r2r_edge_indices[:, jnp.arange(num_r2r_edges), :],
        r2r_edge_domains=m.r2r_edge_domains[:, jnp.arange(num_r2r_edges), :],
        r2p_edge_indices=m.r2p_edge_indices[:, jnp.arange(num_r2p_edges), :] if (m.r2p_edge_indices is not None) else None,
      )

    # Concatenate all padded graph sets and store them
    # NOTE: This line duplicates the memory needed for storing the graphs
    # TODO: Make the concatenation memory efficient
    ## One way is to add another loop and write the graphs one by one in-place in the concatenated array
    self.rigs = tree.tree_map(lambda *v: jnp.concatenate(v), *metadata)

  def _get_sample(self, idx: int) -> Mapping[str, DiscretizedFunction]:
    """Fetches a single sample from the dataset."""
    # NOTE: The arrays are expected to have the following dimensions: [sample, time, variable, space]
    # NOTE: In the rest of the codes, we deal with the following order: [sample, time, space, variable]
    # NOTE: Function values (including BC function) can evolve over time, but coordinates and segments are the same at all times

    # TODO: Support time-dependent datasets

    # Load the coordinates
    # TODO: Avoid loading every time when the geometry is fixed
    if self.metadata.fix:
      x = self.reader[self.metadata.x.path][:]
    else:
      x = self.reader[self.metadata.x.path][idx][:]
      x = x[None, :, :, :]
    # NOTE: Assuming the same coordinates at all times
    assert x.shape[1] == 1
    x = np.swapaxes(x, 2, 3)

    # Load the times
    if self.time_dependent:
      if self.metadata.t is not None:
        t = self.reader[self.metadata.t.path][idx]
        t = t[None, :, None, None]
      else:
        # TODO: Implement
        ...
        # # Define uniform time discretization
        # t = np.linspace(*self.metadata.bbox_t, u.shape[1], endpoint=True)
        # t = t.reshape(1, -1, 1, 1)
    else:
      t = None

    # Set the subsampling permutation
    # TODO: Change self.key every time
    permutation = jax.random.permutation(self.key, x.shape[2])
    _x_size_original = x.shape[2]
    _x_size_after = int(self.metadata.shape[2] / self.space_downsample_factor)
    x = subsample_array(x, permutation, size=_x_size_after, ax=2)

    # Downsample the time axis
    if self.time_dependent:
      if self.time_cutoff_idx:
        t = t[:, :self.time_cutoff_idx]
      if self.time_downsample_factor > 1:
        t = t[:, ::self.time_downsample_factor]

    # Load the registered variables
    functions = {name: None for name in self.metadata.functions.keys()}
    for name, group in self.metadata.functions.items():
      # Get each array and index it if necessary
      arrays = []
      for arr in group.arrays:
        # NOTE: A 4-dimensional array [sample, time, channels, position] is expected
        array: Array = self.reader[arr.path][idx, :, arr.indices if (arr.indices is not None) else slice(None)]
        if len(array.shape) == 2:
          # NOTE: Handle arrays of arrays with variable sizes in space
          array = np.concatenate(array.flatten()).reshape(*array.shape, -1)
        array = array.swapaxes(-2, -1)
        arrays.append(array)
      # Concatenate channels together
      arrays = np.concatenate(arrays, axis=-1)
      # Load the x indices and create a mask accordingly
      mask = np.zeros(shape=(1, 1, _x_size_original), dtype=bool)
      values = np.zeros(shape=(1, 1, _x_size_original, arrays.shape[-1]), dtype=arrays.dtype)
      if group.x_indices is not None:
        x_indices = self.reader[group.x_indices][idx]
        # NOTE: Assuming the same coordinate indices at all times
        assert x_indices.shape[0] == 1
        assert x_indices.shape[1] == 1
        x_indices = x_indices[0, 0]
      else:
        x_indices = np.arange(_x_size_original)
      assert x_indices.shape[0] == arrays.shape[1], f'{x_indices.shape} and {arrays.shape}'
      mask[:, :, x_indices] = True
      values[:, :, x_indices, :] = arrays
      # Permute and subsample the coordinates
      mask = subsample_array(mask, permutation, size=_x_size_after, ax=2)
      values = subsample_array(values, permutation, size=_x_size_after, ax=2)
      # Downsamole and cut the time axis
      if self.time_dependent:
        if self.time_cutoff_idx:
          mask = mask[:, :self.time_cutoff_idx]
          values = values[:, :self.time_cutoff_idx]
        if self.time_downsample_factor > 1:
          mask = mask[:, ::self.time_downsample_factor]
          values = values[:, ::self.time_downsample_factor]

      # Add the variable group
      functions[name] = DiscretizedFunction(mask=mask, values=values)

    return x, t, functions

  def _get_batch(self, idx: Sequence[int], get_graphs: bool = True) -> Batch:
    """Fetches a sample from the dataset, given its global index."""

    # Instantiate the containers
    x = []
    t = [] if self.time_dependent else None
    functions: Mapping[str, Sequence[DiscretizedFunction]] = {name: [] for name in self.metadata.functions.keys()}
    # Get samples one by one
    for _idx in idx:
      _x, _t, _variables = self._get_sample(_idx)
      x.append(_x)
      if self.time_dependent: t.append(_t)
      for name in functions.keys():
        functions[name].append(_variables[name])
    for name in functions.keys():
      functions[name] = DiscretizedFunction(
        mask=np.concatenate([f.mask for f in functions[name]], axis=0),
        values=np.concatenate([f.values for f in functions[name]], axis=0),
      )

    # Stack all arrays
    x = np.concatenate(x, axis=0)
    if self.time_dependent: t = np.concatenate(t, axis=0)

    # Get graphs
    # TODO: Implement with the new structure
    # if (self.rigs is not None) and get_graphs:
    #   g = tree.tree_map(lambda v: v[idx], self.rigs)
    # else:
    #   g = None

    batch = Batch(x=x, t=t, g=None, functions=functions)

    return batch

  def batches(self, split: int, batch_size: int, get_graphs: bool = True, key: PRNGKey = None):
    split_length = self.splits[split][1] - self.splits[split][0]
    assert batch_size > 0
    assert batch_size <= split_length

    _idx_mode = self.splits[split][0] + np.arange(split_length)
    if key is not None:
      _idx_mode = jax.random.permutation(key, _idx_mode)

    len_dividable = split_length - (split_length % batch_size)
    for idx in np.split(_idx_mode[:len_dividable], len_dividable // batch_size):
      batch = self._get_batch(idx, get_graphs=get_graphs)
      yield batch

    if (split_length % batch_size):
      idx = _idx_mode[len_dividable:]
      batch = self._get_batch(idx, get_graphs=get_graphs)
      yield batch

  def __len__(self):
    return self.metadata.shape[0]

def subsample_array(arr, permutation, size, ax):
  arr = np.swapaxes(arr, 0, ax)
  arr = arr[permutation]
  arr = arr[:size]
  arr = np.swapaxes(arr, 0, ax)
  return arr

def load_h5py_group_as_dictionary(group, splits):
  out = {
    key: (
      np.concatenate([group[key][slice(*split)] for split in splits], axis=0)
      if len(group[key].shape) > 0 else group[key]
    )
    if isinstance(group[key], h5py.Dataset)
    else load_h5py_group_as_dictionary(group[key], splits=splits)
    for key in group.keys()
  }
  return out

def flatten_nested_dictionaries(d: dict) -> dict:
  out = {}
  for key, val in d.items():
    if isinstance(val, dict):
      for subkey, subval in flatten_nested_dictionaries(val).items():
        out['/'.join([key, subkey])] = subval
    else:
      out[key] = val

  return out
