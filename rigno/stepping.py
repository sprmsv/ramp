from abc import ABC, abstractmethod

import flax.typing
import jax
import jax.numpy as jnp

from rigno.models.rigno import AbstractOperator
from rigno.utils import Array, normalize, unnormalize


class Stepper(ABC):

  def __init__(self, operator: AbstractOperator):
    self._apply_operator = operator.apply

  def normalize_inputs(self, stats, u_inp, t_inp, tau):
    u_inp_nrm = normalize(
      u_inp,
      shift=stats['trj']['mean'],
      scale=stats['trj']['std'],
    )
    t_inp_nrm = t_inp / stats['time']['max']
    tau_nrm = tau / stats['time']['max']

    return u_inp_nrm, t_inp_nrm, tau_nrm

  @abstractmethod
  def apply(self,
    variables,
    stats,
    u_inp: Array,
    t_inp: Array,
    tau: Array,
    key: flax.typing.PRNGKey = None,
  ):
    """
    Normalizes raw inputs and applies the operator on it.

    t_inp is the time of the input and must be a non-negative integer.
    tau is the time difference and must be an integer greater than zero.
    """
    pass

  def unroll(self,
    variables,
    stats,
    u_inp: Array,
    t_inp: Array,
    tau: Array,
    num_steps: int,
    key: flax.typing.PRNGKey = None,
  ):
    """Apply the stepper multiple times to reach t_inp+tau by dividing tau."""

    def scan_fn_fractional(carry, forcing):
      u_inp, t_inp = carry
      tau = forcing
      u_out = self.apply(
        variables,
        stats,
        u_inp=u_inp,
        t_inp=t_inp,
        tau=tau,
        key=key,
      )
      u_next = u_out
      t_next = t_inp + tau
      carry = (u_next, t_next)
      return carry, u_out

    tau_tiled = jnp.repeat(tau, repeats=num_steps)
    tau_fract = tau_tiled / num_steps
    forcing = tau_fract
    (u_out, _), _ = jax.lax.scan(f=scan_fn_fractional,
      init=(u_inp, t_inp), xs=forcing, length=num_steps)

    return u_out

  @abstractmethod
  def get_loss_inputs(self,
    variables,
    stats,
    u_inp: Array,
    t_inp: Array,
    u_tgt: Array,
    tau: Array,
    key: flax.typing.PRNGKey = None,
  ):
    """
    Calculates prediction and target variables, ready to be given as input to the loss function.

    t_inp is the time of the input and must be a non-negative integer.
    tau is the time difference and must be an integer greater than zero.
    """
    pass

  def get_intermediates(self,
    variables,
    stats,
    u_inp: Array,
    t_inp: Array,
    tau: Array,
    key: flax.typing.PRNGKey = None,
  ):
    # Normalize inputs
    u_inp_nrm, t_inp_nrm, tau_nrm = self.normalize_inputs(stats, u_inp, t_inp, tau)

    # Get predicted normalized derivatives
    _, state = self._apply_operator(
      variables,
      u_inp=u_inp_nrm,
      t_inp=t_inp_nrm,
      tau=tau_nrm,
      key=key,
      capture_intermediates=True,
    )

    return state['intermediates']

class TimeDerivativeUpdater(Stepper):

  def apply(self,
    variables,
    stats,
    u_inp: Array,
    t_inp: Array,
    tau: Array,
    key: flax.typing.PRNGKey = None,
  ):
    """
    Normalizes raw inputs and applies the operator on it.

    t_inp is the time of the input and must be a non-negative integer.
    tau is the time difference and must be an integer greater than zero.
    """

    # Normalize inputs
    u_inp_nrm, t_inp_nrm, tau_nrm = self.normalize_inputs(stats, u_inp, t_inp, tau)

    # Get predicted normalized derivatives
    d_prd_nrm = self._apply_operator(
      variables,
      u_inp=u_inp_nrm,
      t_inp=t_inp_nrm,
      tau=tau_nrm,
      key=key,
    )

    # Unnormalize predicted derivatives
    d_prd = unnormalize(
      d_prd_nrm,
      mean=stats['der']['mean'],
      std=stats['der']['std'],
    )

    # Get predicted output
    u_prd = u_inp + (d_prd * tau)

    return u_prd

  def get_loss_inputs(self,
    variables,
    stats,
    u_inp: Array,
    t_inp: Array,
    u_tgt: Array,
    tau: Array,
    key: flax.typing.PRNGKey = None,
  ):
    """
    Calculates prediction and target variables, ready to be given as input to the loss function.

    t_inp is the time of the input and must be a non-negative integer.
    tau is the time difference and must be an integer greater than zero.
    """

    # Normalize inputs
    u_inp_nrm, t_inp_nrm, tau_nrm = self.normalize_inputs(stats, u_inp, t_inp, tau)

    # Get predicted normalized derivatives
    d_prd_nrm = self._apply_operator(
      variables,
      u_inp=u_inp_nrm,
      t_inp=t_inp_nrm,
      tau=tau_nrm,
      key=key,
    )

    # Get target normalized derivatives
    d_tgt = (u_tgt - u_inp) / tau
    d_tgt_nrm = normalize(
      d_tgt,
      shift=stats['der']['mean'],
      scale=stats['der']['std'],
    )

    return (d_tgt_nrm, d_prd_nrm)

class ResidualUpdater(Stepper):

  def apply(self,
    variables,
    stats,
    u_inp: Array,
    t_inp: Array,
    tau: Array,
    key: flax.typing.PRNGKey = None,
  ):
    """
    Normalizes raw inputs and applies the operator on it.

    t_inp is the time of the input and must be a non-negative integer.
    tau is the time difference and must be an integer greater than zero.
    """

    # Normalize inputs
    u_inp_nrm, t_inp_nrm, tau_nrm = self.normalize_inputs(stats, u_inp, t_inp, tau)

    # Get predicted normalized derivative
    r_prd_nrm = self._apply_operator(
      variables,
      u_inp=u_inp_nrm,
      t_inp=t_inp_nrm,
      tau=tau_nrm,
      key=key,
    )

    # Unnormalize predicted residuals
    r_prd = unnormalize(
      r_prd_nrm,
      mean=stats['res']['mean'],
      std=stats['res']['std'],
    )

    # Get predicted output
    u_prd = u_inp + r_prd

    return u_prd

  def get_loss_inputs(self,
    variables,
    stats,
    u_inp: Array,
    t_inp: Array,
    u_tgt: Array,
    tau: Array,
    key: flax.typing.PRNGKey = None,
  ):
    """
    Calculates prediction and target variables, ready to be given as input to the loss function.

    t_inp is the time of the input and must be a non-negative integer.
    tau is the time difference and must be an integer greater than zero.
    """

    # Normalize inputs
    u_inp_nrm, t_inp_nrm, tau_nrm = self.normalize_inputs(stats, u_inp, t_inp, tau)

    # Get predicted normalized residuals
    r_prd_nrm = self._apply_operator(
      variables,
      u_inp=u_inp_nrm,
      t_inp=t_inp_nrm,
      tau=tau_nrm,
      key=key,
    )

    # Get target normalized residuals
    r_tgt = u_tgt - u_inp
    r_tgt_nrm = normalize(
      r_tgt,
      shift=stats['res']['mean'],
      scale=stats['res']['std'],
    )

    return (r_tgt_nrm, r_prd_nrm)

class OutputUpdater(Stepper):

  def apply(self,
    variables,
    stats,
    u_inp: Array,
    t_inp: Array,
    tau: Array,
    key: flax.typing.PRNGKey = None,
  ):
    """
    Normalizes raw inputs and applies the operator on it.

    t_inp is the time of the input and must be a non-negative integer.
    tau is the time difference and must be an integer greater than zero.
    """

    # Normalize inputs
    u_inp_nrm, t_inp_nrm, tau_nrm = self.normalize_inputs(stats, u_inp, t_inp, tau)

    # Get predicted normalized output
    u_prd_nrm = self._apply_operator(
      variables,
      u_inp=u_inp_nrm,
      t_inp=t_inp_nrm,
      tau=tau_nrm,
      key=key,
    )

    # Unnormalize predicted output
    u_prd = unnormalize(
      u_prd_nrm,
      mean=stats['trj']['mean'],
      std=stats['trj']['std'],
    )

    return u_prd

  def get_loss_inputs(self,
    variables,
    stats,
    u_inp: Array,
    t_inp: Array,
    u_tgt: Array,
    tau: Array,
    key: flax.typing.PRNGKey = None,
  ):
    """
    Calculates prediction and target variables, ready to be given as input to the loss function.

    t_inp is the time of the input and must be a non-negative integer.
    tau is the time difference and must be an integer greater than zero.
    """

    # Normalize inputs
    u_inp_nrm, t_inp_nrm, tau_nrm = self.normalize_inputs(stats, u_inp, t_inp, tau)

    # Get predicted normalized output
    u_prd_nrm = self._apply_operator(
      variables,
      u_inp=u_inp_nrm,
      t_inp=t_inp_nrm,
      tau=tau_nrm,
      key=key,
    )

    # Get target normalized output
    u_tgt_nrm = normalize(
      u_tgt,
      shift=stats['trj']['mean'],
      scale=stats['trj']['std'],
    )

    return (u_tgt_nrm, u_prd_nrm)

class AutoregressiveStepper:

  def __init__(self, stepper: Stepper, tau_max: float = 1., tau_base: float = 1.):
    """
    Class for autoregressive inferrence of an operator.

    Args:
        stepper: Uses an operator with proper stepping method.
        tau_max: Maximum time delta of direct predictions with respect to the time resolution of the trained model. Defaults to 1.
        tau_base: Time resolution of the output with respect to the time resolution of the trained model. Defaults to 1.
    """

    # FIXME: Maybe we can benefit from checkpointing scan_fn instead
    self.tau_base = tau_base
    if tau_max >= tau_base:
      assert tau_max % tau_base == 0
      self.num_steps_direct = int(tau_max / tau_base)
      self._apply_operator = jax.checkpoint(stepper.apply)
    else:
      assert tau_base % tau_max == 0
      self.num_steps_direct = 1
      num_unrolls_per_step = int(tau_base / tau_max)
      def _stepper_unroll(*args, **kwargs):
        return stepper.unroll(*args, **kwargs, num_steps=num_unrolls_per_step)
      self._apply_operator = jax.checkpoint(_stepper_unroll)

  def unroll(self,
    variables: flax.typing.VariableDict,
    stats: flax.typing.VariableDict,
    u_inp: Array,
    t_inp: Array,
    num_steps: int,
    key: flax.typing.PRNGKey = None,
  ) -> Array:

    batch_size = u_inp.shape[0]
    num_grid_nodes = u_inp.shape[2:4]
    num_outputs = u_inp.shape[-1]
    t_inp = t_inp.astype(float)
    random = (key is not None)
    if not random:
      key, _ = jax.random.split(jax.random.PRNGKey(0))  # NOTE: It won't be used

    def scan_fn_direct(carry, forcing):
      u_inp, t_inp = carry
      tau = forcing[0]
      key = forcing[-2:].astype('uint32')
      u_out = self._apply_operator(
        variables,
        stats,
        u_inp=u_inp,
        t_inp=t_inp,
        tau=tau,
        key=(key if random else None),
      )
      carry = (u_inp, t_inp)  # NOTE: The input is the same for all tau
      return carry, u_out

    def scan_fn_autoregressive(carry, forcing):
      u_inp, t_inp = carry
      tau = forcing[:-2]
      key = forcing[-2:].astype('uint32')
      _num_direct_steps = tau.shape[0]
      keys = jax.random.split(key, num=_num_direct_steps)
      forcing = jnp.concatenate([tau.reshape(-1, 1), keys], axis=-1)
      _, u_out = jax.lax.scan(f=scan_fn_direct,
        init=(u_inp, t_inp), xs=forcing, length=_num_direct_steps)
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
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=num_jumps)
    forcings = jnp.concatenate([tau_tiled, keys], axis=-1)
    (u_next, t_next), rollout_full = jax.lax.scan(
      f=scan_fn_autoregressive,
      init=(u_inp, t_inp),
      xs=forcings,
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
      key, subkey = jax.random.split(key, num=2)
      keys = subkey.reshape(1, 2)
      forcings = jnp.concatenate([tau_tiled, keys], axis=-1)
      (u_next, t_next), rollout_part = jax.lax.scan(
        f=scan_fn_autoregressive,
        init=(u_next, t_next),
        xs=forcings,
        length=1
      )
      rollout_part = rollout_part.swapaxes(0, 1)
      rollout_part = rollout_part.reshape(
        batch_size, num_steps_rem, *num_grid_nodes, num_outputs)
      rollout = jnp.concatenate([rollout, rollout_part], axis=1)

    # Exclude the last timestep because it is returned separately
    rollout = rollout[:, :-1]

    return rollout, u_next

  def jump(self,
    variables: flax.typing.VariableDict,
    stats: flax.typing.VariableDict,
    u_inp: Array,
    t_inp: Array,
    num_jumps: int,
    key: flax.typing.PRNGKey = None,
  ) -> Array:
    """Takes num_jumps large steps, each of length num_steps_direct."""

    t_inp = t_inp.astype(float)
    random = (key is not None)
    if not random:
      key, _ = jax.random.split(jax.random.PRNGKey(0))  # Won't be used

    def scan_fn(carry, forcing):
      u_inp, t_inp = carry
      subkey = forcing if random else None
      tau = self.tau_base * self.num_steps_direct
      u_out = self._apply_operator(
        variables,
        stats,
        u_inp=u_inp,
        t_inp=t_inp,
        tau=tau,
        key=subkey,
      )
      u_inp_next = u_out
      t_inp_next = t_inp + tau
      carry = (u_inp_next, t_inp_next)
      rollout = None
      return carry, rollout

    keys = jax.random.split(key, num=num_jumps)
    forcings = keys
    (u_next, t_next), _ = jax.lax.scan(
      f=scan_fn,
      init=(u_inp, t_inp),
      xs=forcings,
      length=num_jumps,
    )

    return u_next
