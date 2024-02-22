from absl import app
from absl import flags
from absl import logging
import jax

from typing import Tuple, Any
from time import time

from pathlib import Path
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
import matplotlib.pyplot as plt
import numpy as np
from graphneuralpdesolver.models.graphneuralpdesolver import GraphNeuralPDESolver

from graphneuralpdesolver.dataset import read_datasets, prepare_dataset

from utils import disable_logging


FLAGS = flags.FLAGS
# flags.DEFINE_string(name='', default=None, help='', required=False)





def main(args):
  if len(args) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with disable_logging():
    process_index = jax.process_index()
    process_count = jax.process_count()
    local_devices = jax.local_devices()
  logging.info('JAX host: %d / %d', process_index, process_count)
  logging.info('JAX local devices: %r', local_devices)


if __name__ == '__main__':
  logging.set_verbosity('info')
  app.run(main)
