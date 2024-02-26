from absl import logging
from typing import Union

import numpy as np
import jax.numpy as jnp


Array = Union[jnp.ndarray, np.ndarray]

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
