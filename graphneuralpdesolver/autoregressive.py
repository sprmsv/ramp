import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.typing

from graphneuralpdesolver.utils import Array
from graphneuralpdesolver.models.graphneuralpdesolver import AbstractPDESolver


class AutoregressivePredictor:

  def __init__(self, predictor: AbstractPDESolver):
    self._predictor = predictor

  def __call__(self, variables: flax.typing.VariableDict,
               u_inp: Array, specs: Array, num_steps: int) -> Array:

    num_times_input = self._predictor.num_times_input
    num_times_output = self._predictor.num_times_output
    batch_size = u_inp.shape[0]
    assert u_inp.shape[1] == num_times_input
    num_grid_nodes = u_inp.shape[2]
    num_outputs = u_inp.shape[3]

    def scan_fn(u_inp, forcing):
      u_out = self._predictor.apply(variables, u_inp=u_inp, specs=specs)
      if num_times_input > num_times_output:
        u_inp_next = jnp.concatenate([u_inp[:, -(num_times_input-num_times_output):], u_out], axis=1)
      else:
        u_inp_next = u_out[:, -num_times_input:]
      return u_inp_next, u_out

    # CHECK: Use flax.linen.scan for the for loops?
    forcings = None
    _, rollout = jax.lax.scan(f=scan_fn, init=u_inp, xs=forcings, length=num_steps)
    rollout = rollout.swapaxes(0, 1).reshape(batch_size, num_steps*num_times_output, num_grid_nodes, num_outputs)

    return rollout
