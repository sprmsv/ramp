import jax
import jax.numpy as jnp
import flax.typing

from graphneuralpdesolver.utils import Array
from graphneuralpdesolver.models.graphneuralpdesolver import AbstractOperator


class AutoregressivePredictor:

  def __init__(self, operator: AbstractOperator, num_steps_direct: int = 1):
    # FIXME: Maybe we can benefit from checkpointing scan_fn instead
    self._apply_operator = jax.checkpoint(operator.apply)
    self.num_steps_direct = num_steps_direct

  def __call__(self, variables: flax.typing.VariableDict,
    specs: Array, u_inp: Array, num_steps: int) -> Array:

    batch_size = u_inp.shape[0]
    num_grid_nodes = u_inp.shape[2]
    num_outputs = u_inp.shape[3]

    time_deltas = (1. + jnp.arange(self.num_steps_direct)).reshape(-1, 1)

    def scan_fn_direct(u_inp, dt):
      u_out = self._apply_operator(variables, specs=specs, u_inp=u_inp, dt=dt)
      return u_out, u_inp

    def scan_fn_autoregressive(u_inp, forcing):
      u_next, u_out = jax.lax.scan(f=scan_fn_direct,
        init=u_inp, xs=time_deltas, length=self.num_steps_direct)
      u_out = jnp.squeeze(u_out, axis=2).swapaxes(0, 1)
      return u_next, u_out

    forcings = None
    u_next, rollout = jax.lax.scan(f=scan_fn_autoregressive,
      init=u_inp, xs=forcings, length=(num_steps // self.num_steps_direct))
    rollout = rollout.swapaxes(0, 1)
    rollout = rollout.reshape(batch_size, num_steps, num_grid_nodes, num_outputs)

    return rollout, u_next

  def jump(self, variables: flax.typing.VariableDict,
    specs: Array, u_inp: Array, num_steps: int) -> Array:

    dt = jnp.array(self.num_steps_direct).reshape(1,)

    def scan_fn(u_inp, forcing):
      u_out = self._apply_operator(variables, specs=specs, u_inp=u_inp, dt=dt)
      u_inp_next = u_out
      rollout = None
      return u_inp_next, rollout

    # CHECK: Use flax.linen.scan for the for loops?
    forcings = None
    u_next, _ = jax.lax.scan(
      f=scan_fn, init=u_inp, xs=forcings, length=(num_steps // self.num_steps_direct))

    return u_next
