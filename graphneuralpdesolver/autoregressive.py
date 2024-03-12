import jax
import jax.numpy as jnp
import flax.typing

from graphneuralpdesolver.utils import Array
from graphneuralpdesolver.models.graphneuralpdesolver import AbstractOperator
from graphneuralpdesolver.dataset import normalize, unnormalize


class OperatorNormalizer:

  def __init__(self, operator: AbstractOperator, stats_trj, stats_res):
    self._apply_operator = operator.apply
    self._stats_trj_mean, self._stats_trj_std = stats_trj
    self._stats_res_mean, self._stats_res_std = stats_res

  def apply(self, variables, specs: jnp.ndarray, u_inp: jnp.ndarray, ndt: int):
    # ndt >= 1
    # ndt_inp >= 0
    # TODO: START FROM ZERO

    # Get normalized predicted residuals
    u_inp_nrm = normalize(
      u_inp,
      mean=self._stats_trj_mean,
      std=self._stats_trj_std
    )
    # TODO: Normalize specifications as well
    r_prd_nrm = self._apply_operator(
      variables,
      specs=specs,
      u_inp=u_inp_nrm,
      ndt=jnp.array(ndt).astype(jnp.float32).reshape(1,),
    )

    # Get predicted output
    r_prd = unnormalize(
      r_prd_nrm,
      mean=self._stats_res_mean[(ndt-1)],
      std=self._stats_res_std[(ndt-1)],
    )
    u_prd = u_inp + r_prd

    return u_prd

  def get_loss_inputs(self, variables, specs: jnp.ndarray, u_inp: jnp.ndarray, u_tgt: jnp.ndarray, ndt: int):
    # ndt >= 1
    # ndt_inp >= 0
    # TODO: START FROM ZERO

    # Get normalized predicted residuals
    u_inp_nrm = normalize(
      u_inp, mean=self._stats_trj_mean, std=self._stats_trj_std)
    r_prd_nrm = self._apply_operator(
      variables,
      specs=specs,
      u_inp=u_inp_nrm,
      ndt=jnp.array([1], dtype=jnp.float32),
    )

    # Get normalized target residuals
    r_tgt = u_tgt - u_inp
    r_tgt_nrm = normalize(
      r_tgt,
      mean=self._stats_res_mean[(ndt-1)],
      std=self._stats_res_std[(ndt-1)],
    )

    return (r_prd_nrm, r_tgt_nrm)

class AutoregressivePredictor:

  def __init__(self, operator: AbstractOperator, num_steps_direct: int = 1):
    # FIXME: Maybe we can benefit from checkpointing scan_fn instead
    self._apply_operator = jax.checkpoint(operator.apply)
    self.num_steps_direct = num_steps_direct

  def __call__(self, variables: flax.typing.VariableDict,
    specs: Array, u_inp: Array, num_jumps: int) -> Array:

    batch_size = u_inp.shape[0]
    num_grid_nodes = u_inp.shape[2]
    num_outputs = u_inp.shape[3]

    time_deltas = (1. + jnp.arange(self.num_steps_direct)).reshape(-1, 1)

    def scan_fn_direct(u_inp, ndt):
      u_out = self._apply_operator(variables, specs=specs, u_inp=u_inp, ndt=ndt)
      return u_inp, u_out

    def scan_fn_autoregressive(u_inp, forcing):
      _, u_out = jax.lax.scan(f=scan_fn_direct,
        init=u_inp, xs=time_deltas, length=self.num_steps_direct)
      u_out = jnp.squeeze(u_out, axis=2).swapaxes(0, 1)
      u_next = u_out[:, -1:]
      return u_next, u_out

    forcings = None
    u_next, rollout = jax.lax.scan(f=scan_fn_autoregressive,
      init=u_inp, xs=forcings, length=num_jumps)
    rollout = rollout.swapaxes(0, 1)
    rollout = rollout.reshape(
      batch_size, (num_jumps*self.num_steps_direct), num_grid_nodes, num_outputs)
    rollout = jnp.concatenate([u_inp, rollout[:, :-1]], axis=1)

    return rollout, u_next

  def jump(self, variables: flax.typing.VariableDict,
    specs: Array, u_inp: Array, num_jumps: int) -> Array:
    """Takes num_jumps large steps, each of length num_steps_direct."""

    ndt = jnp.array(self.num_steps_direct).reshape(1,)

    def scan_fn(u_inp, forcing):
      u_out = self._apply_operator(variables, specs=specs, u_inp=u_inp, ndt=ndt)
      u_inp_next = u_out
      rollout = None
      return u_inp_next, rollout

    # CHECK: Use flax.linen.scan for the for loops?
    forcings = None
    u_next, _ = jax.lax.scan(f=scan_fn,
      init=u_inp, xs=forcings, length=num_jumps)

    return u_next
