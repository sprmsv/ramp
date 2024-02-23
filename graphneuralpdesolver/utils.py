from absl import logging
from typing import Union

import numpy as np
import jax.numpy as jnp


Array = Union[jnp.ndarray, np.ndarray]

class disable_logging:
  """TODO: Write"""

  def __init__(self):
    self.level = None

  def __enter__(self):
    self.level = logging.get_verbosity()
    logging.set_verbosity(-1)

  def __exit__(self, exc_type, exc_value, traceback):
    logging.set_verbosity(self.level)
