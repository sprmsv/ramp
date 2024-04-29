import jax
import jax.numpy as jnp
import flax.typing

from graphneuralpdesolver.utils import Array, normalize, unnormalize
from graphneuralpdesolver.models.graphneuralpdesolver import AbstractOperator


class OperatorNormalizer:

  def __init__(self, operator: AbstractOperator):
    self._apply_operator = operator.apply

  def apply(self, variables, stats, specs: Array, u_inp: Array, t_inp: Array, tau: int):
    """
    Normalizes raw inputs and applies the operator on it.

    t_inp is the time of the input and must be a non-negative integer.
    tau is the time difference and must be an integer greater than zero.
    """

    # Get the corresponding the global statistics
    # NOTE: Swap axes is necessary because we will have statistics of shape (1, bsz, ...)
    # NOTE:   and we want shape (bsz, 1, ...), same as u_inp
    # NOTE: The trick here is that each sample in the batch gets the statistics of its corresponding t_inp
    stats_trj_mean = stats['trj']['mean'][:, t_inp.reshape(-1)].swapaxes(0, 1)
    stats_trj_std = stats['trj']['std'][:, t_inp.reshape(-1)].swapaxes(0, 1)
    stats_res_mean = stats['res']['mean'][(tau-1)][:, t_inp.reshape(-1)].swapaxes(0, 1)
    stats_res_std = stats['res']['std'][(tau-1)][:, t_inp.reshape(-1)].swapaxes(0, 1)

    # Get normalized predicted residuals
    # TODO: Normalize specs as well
    u_inp_nrm = normalize(
      u_inp,
      mean=stats_trj_mean,
      std=stats_trj_std,
    )
    r_prd_nrm = self._apply_operator(
      variables,
      specs=specs,
      u_inp=u_inp_nrm,
      t_inp=t_inp,
      tau=tau,
    )

    # Get predicted output
    r_prd = unnormalize(
      r_prd_nrm,
      mean=stats_res_mean,
      std=stats_res_std,
    )
    u_prd = u_inp + r_prd

    return u_prd

  def get_loss_inputs(self, variables, stats, specs: Array, u_inp: Array, t_inp: Array, u_tgt: Array, tau: int):
    """
    Calculates prediction and target variables, ready to be given as input to the loss function.

    t_inp is the time of the input and must be a non-negative integer.
    tau is the time difference and must be an integer greater than zero.
    """

    # Get the corresponding the global statistics
    # NOTE: Swap axes is necessary because we will have statistics of shape (1, bsz, ...)
    # NOTE:   and we want shape (bsz, 1, ...), same as u_inp
    # NOTE: The trick here is that each sample in the batch gets the statistics of its corresponding t_inp
    stats_trj_mean = stats['trj']['mean'][:, t_inp.reshape(-1)].swapaxes(0, 1)
    stats_trj_std = stats['trj']['std'][:, t_inp.reshape(-1)].swapaxes(0, 1)
    stats_res_mean = stats['res']['mean'][(tau-1)][:, t_inp.reshape(-1)].swapaxes(0, 1)
    stats_res_std = stats['res']['std'][(tau-1)][:, t_inp.reshape(-1)].swapaxes(0, 1)

    # TODO: Normalize t_inp and tau

    # Get normalized predicted residuals
    u_inp_nrm = normalize(
      u_inp,
      mean=stats_trj_mean,
      std=stats_trj_std,
    )
    r_prd_nrm = self._apply_operator(
      variables,
      specs=specs,
      u_inp=u_inp_nrm,
      t_inp=t_inp,
      tau=tau,
    )

    # Get normalized target residuals
    r_tgt = u_tgt - u_inp
    r_tgt_nrm = normalize(
      r_tgt,
      mean=stats_res_mean,
      std=stats_res_std,
    )

    return (r_prd_nrm, r_tgt_nrm)

class AutoregressivePredictor:

  def __init__(self, normalizer: OperatorNormalizer, num_steps_direct: int = 1, tau_base: int = 1):
    # FIXME: Maybe we can benefit from checkpointing scan_fn instead
    self._apply_operator = jax.checkpoint(normalizer.apply)
    self.num_steps_direct = num_steps_direct
    self.tau_base = tau_base

  def unroll(self, variables: flax.typing.VariableDict, stats: flax.typing.VariableDict,
    specs: Array, u_inp: Array, t_inp: Array, num_steps: int) -> Array:

    batch_size = u_inp.shape[0]
    num_grid_nodes = u_inp.shape[2:4]
    num_outputs = u_inp.shape[-1]

    def scan_fn_direct(carry, _tau):
      u_inp, t_inp = carry
      u_out = self._apply_operator(variables, stats, specs=specs, u_inp=u_inp, t_inp=t_inp, tau=_tau)
      carry = (u_inp, t_inp)  # NOTE: the input is the same for all _tau
      return carry, u_out

    def scan_fn_autoregressive(carry, forcing):
      u_inp, t_inp = carry
      _, u_out = jax.lax.scan(f=scan_fn_direct,
        init=(u_inp, t_inp), xs=forcing, length=forcing.shape[0])
      u_out = jnp.squeeze(u_out, axis=2).swapaxes(0, 1)
      u_next = u_out[:, -1:]
      t_next = t_inp + self.tau_base * self.num_steps_direct
      carry = (u_next, t_next)
      return carry, u_out

    # Get full sets of direct predictions
    num_jumps = num_steps // self.num_steps_direct
    tau_tiled = self.tau_base * jnp.tile(
      jnp.arange(1, self.num_steps_direct+1), reps=(num_jumps, 1)
    ).reshape(num_jumps, self.num_steps_direct)
    (u_next, t_next), rollout_full = jax.lax.scan(
      f=scan_fn_autoregressive,
      init=(u_inp, t_inp),
      xs=tau_tiled,
      length=num_jumps,
    )
    rollout_full = rollout_full.swapaxes(0, 1)
    rollout_full = rollout_full.reshape(
      batch_size, (num_jumps*self.num_steps_direct), *num_grid_nodes, num_outputs)
    rollout = jnp.concatenate([u_inp, rollout_full], axis=1)

    # Get the last set of direct predictions partially (if necessary)
    num_steps_rem = num_steps % self.num_steps_direct
    if num_steps_rem:
      tau_tiled = self.tau_base * jnp.arange(1, num_steps_rem+1).reshape(1, num_steps_rem)
      (u_next, t_next), rollout_part = jax.lax.scan(
        f=scan_fn_autoregressive,
        init=(u_next, t_next),
        xs=tau_tiled,
        length=1
      )
      rollout_part = rollout_part.swapaxes(0, 1)
      rollout_part = rollout_part.reshape(
        batch_size, num_steps_rem, *num_grid_nodes, num_outputs)
      rollout = jnp.concatenate([rollout, rollout_part], axis=1)

    # Exclude the last timestep because it is returned separately
    rollout = rollout[:, :-1]

    return rollout, u_next

  def jump(self, variables: flax.typing.VariableDict, stats: flax.typing.VariableDict,
    specs: Array, u_inp: Array, t_inp: Array, num_jumps: int) -> Array:
    """Takes num_jumps large steps, each of length num_steps_direct."""

    def scan_fn(carry, forcing):
      u_inp, t_inp = carry
      tau = self.tau_base * self.num_steps_direct
      u_out = self._apply_operator(
        variables,
        stats,
        specs=specs,
        u_inp=u_inp,
        t_inp=t_inp,
        tau=tau,
      )
      u_inp_next = u_out
      t_inp_next = t_inp + tau
      carry = (u_inp_next, t_inp_next)
      rollout = None
      return carry, rollout

    # CHECK: Use flax.linen.scan for the for loops?
    forcings = None
    (u_next, t_next), _ = jax.lax.scan(f=scan_fn,
      init=(u_inp, t_inp), xs=forcings, length=num_jumps)

    return u_next
