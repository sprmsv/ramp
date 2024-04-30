from absl import logging
from typing import Union, Sequence

import numpy as np
import jax
import jax.numpy as jnp
import flax.typing


Array = Union[jnp.ndarray, np.ndarray]
ScalarArray = Union[jnp.ndarray, np.ndarray]

class disable_logging:
  """Context manager for disabling the logging."""

  def __init__(self, level: int = -1):
    self.level_context = level
    self.level_init = None

  def __enter__(self):
    self.level_init = logging.get_verbosity()
    logging.set_verbosity(self.level_context)

  def __exit__(self, exc_type, exc_value, traceback):
    logging.set_verbosity(self.level_init)

def shuffle_arrays(key: flax.typing.PRNGKey, arrays: Sequence[Array]) -> Sequence[Array]:
  """Shuffles a set of arrays with the same random permutation along the first axis."""

  size = arrays[0].shape[0]
  assert all([arr.shape[0] == size for arr in arrays])
  permutation = jax.random.permutation(key, size)

  return [arr[permutation] for arr in arrays]


def normalize(arr: Array, shift: Array, scale: Array):
  scale = jnp.where(scale == 0., 1., scale)
  arr = (arr - shift) / scale
  return arr

def unnormalize(arr: Array, mean: Array, std: Array):
  arr = std * arr + mean
  return arr

def calculate_fd_derivative(arr, axes):
  grads = []
  for ax in axes:
    grads.append((jnp.roll(arr, axis=ax, shift=-1) - jnp.roll(arr, axis=ax, shift=1)) / 2)
  return (*grads,)
