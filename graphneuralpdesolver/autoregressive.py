import jax
import jax.numpy as jnp
import flax.typing

from graphneuralpdesolver.utils import Array
from graphneuralpdesolver.models.graphneuralpdesolver import AbstractOperator


class AutoregressivePredictor:

  def __init__(self, operator: AbstractOperator, full_rollout: bool = True):
    self._apply_operator = jax.checkpoint(operator.apply)
    self.full_rollout = full_rollout

  def __call__(self, variables: flax.typing.VariableDict, specs: Array, u_inp: Array,
    num_steps: int, num_steps_direct: int = 1) -> Array:

    batch_size = u_inp.shape[0]
    num_grid_nodes = u_inp.shape[2]
    num_outputs = u_inp.shape[3]

    assert num_steps_direct > 0
    assert not (num_steps % num_steps_direct)

    # If num_steps is zero, u_next is u_inp and rollout is empty
    if not num_steps:
      rollout = None
      u_next = u_inp
      return rollout, u_next

    time_deltas = (1. + jnp.arange(num_steps_direct)).reshape(-1, 1)
    if not self.full_rollout:
      time_deltas = time_deltas[-1:]

    def scan_fn(u_inp, forcing):
      u_out = jax.vmap(
        lambda dt: self._apply_operator(variables, specs=specs, u_inp=u_inp, dt=dt)
      )(time_deltas)
      u_out = jnp.squeeze(u_out, axis=2).swapaxes(0, 1)
      u_inp_next = u_out[:, -1:]  # Take the last time step as the next input
      rollout = jnp.concatenate([u_inp, u_out[:, :-1]], axis=1)
      return u_inp_next, rollout

    # CHECK: Use flax.linen.scan for the for loops?
    forcings = None
    u_next, rollout = jax.lax.scan(
      f=scan_fn, init=u_inp, xs=forcings, length=(num_steps // num_steps_direct))
    rollout = rollout.swapaxes(0, 1)
    if not self.full_rollout:
      rollout = rollout.repeat(num_steps_direct, axis=1)
    rollout = rollout.reshape(batch_size, num_steps, num_grid_nodes, num_outputs)

    return rollout, u_next
