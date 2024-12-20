"""Metadata of the datasets."""

from dataclasses import dataclass
from typing import Sequence, Mapping, Tuple


@dataclass
class ArrayMetadata:
  path: str
  indices: Sequence[int] = None
  names: Sequence[str] = None
  signed: Sequence[str] = None

@dataclass
class ArrayGroup:
  arrays: Sequence[ArrayMetadata]
  x_indices: str = None

  @property
  def signed(self):
    out = []
    for arr in self.arrays:
      for idx in arr.indices:
        out.append(arr.signed[idx])
    return out

  @property
  def names(self):
    out = []
    for arr in self.arrays:
      for idx in arr.indices:
        out.append(arr.signed[idx])
    return out

@dataclass
class Metadata:
  periodic: bool
  fix: bool
  shape: Tuple[int, int, int, int]
  x: ArrayMetadata
  t: ArrayMetadata
  bbox_x: Tuple[Tuple[float, float], Tuple[float, float]]
  bbox_t: Tuple[float, float]
  functions: Mapping[str, ArrayGroup]
  inp: Sequence[str]
  out: Sequence[str]

DATASET_METADATA = {
  'poisson-circle-bc1': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 16646, -1),
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'u': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0], names=['$u$'], signed=[True])],
        x_indices=None,
      ),
      'c': ArrayGroup(
        arrays=[
          ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[True]),
          ArrayMetadata(path='interior/source', indices=[0], names=['$f$'], signed=[True]),
        ],
        x_indices=None,
      ),
      'h': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/dirichlet/indices',
      ),
    },
    inp=['c', 'h'],
    out=['u'],
  ),
  'poisson-circle-bc2': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 16646, -1),
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'u': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0], names=['$u$'], signed=[True])],
        x_indices=None,
      ),
      'c': ArrayGroup(
        arrays=[
          ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[True]),
          ArrayMetadata(path='interior/source', indices=[0], names=['$f$'], signed=[True]),
        ],
        x_indices=None,
      ),
      'h': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/robin/g', indices=[0], names=['$g_B$'], signed=[True]),
          ArrayMetadata(path='boundaries/robin/alpha', indices=[0], names=['$\\alpha_R$'], signed=[True]),
        ],
        x_indices='boundaries/robin/indices',
      ),
    },
    inp=['c', 'h'],
    out=['u'],
  ),
  'poisson-circle-bc3': Metadata(
    periodic=False,
    fix=True,
    shape=(-1, 1, 16646, -1),
    x=ArrayMetadata(path='coordinates', indices=[0, 1], names=['$x$, $y$'], signed=[True, True]),
    t=None,
    bbox_x=((-1., -1.), (+1., +1.)),
    bbox_t=None,
    functions={
      'u': ArrayGroup(
        arrays=[ArrayMetadata(path='interior/solution', indices=[0], names=['$u$'], signed=[True])],
        x_indices=None,
      ),
      'c': ArrayGroup(
        arrays=[
          ArrayMetadata(path='interior/sdf', indices=[0], names=['SDF'], signed=[True]),
          ArrayMetadata(path='interior/source', indices=[0], names=['$f$'], signed=[True]),
        ],
        x_indices=None,
      ),
      'h_d': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/dirichlet/g', indices=[0], names=['$g_D$'], signed=[True]),
        ],
        x_indices='boundaries/dirichlet/indices'
      ),
      'h_r': ArrayGroup(
        arrays=[
          ArrayMetadata(path='boundaries/robin/g', indices=[0], names=['$g_B$'], signed=[True]),
          ArrayMetadata(path='boundaries/robin/alpha', indices=[0], names=['$\\alpha_R$'], signed=[True]),
        ],
        x_indices='boundaries/robin/indices',
      ),
    },
    inp=['c', 'h_d', 'h_r'],
    out=['u'],
  ),
}
