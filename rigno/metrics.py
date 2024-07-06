from dataclasses import dataclass
from copy import copy
from typing import Sequence

import jax.numpy as jnp

from rigno.utils import Array, ScalarArray


EPSILON = 1e-10

@dataclass
class BatchMetrics:
    mse: Array = None
    l1: Array = None
    l2: Array = None

    def map(self, f):
        for key in self.__dict__.keys():
            self.__setattr__(key, f(self.__getattribute__(key)))

    def reshape(self, shape):
        self.map(lambda m: m.reshape(shape))

    def __add__(self, obj):
      out = copy(self)
      for key in self.__dict__.keys():
        out.__setattr__(key, self.__getattribute__(key) + obj.__getattribute__(key))
      return out

@dataclass
class Metrics:
    mse: float = None
    l1: float = None
    l2: float = None

@dataclass
class EvalMetrics:
  direct_tau_frac: Metrics = None
  direct_tau_min: Metrics = None
  direct_tau_max: Metrics = None
  rollout: Metrics = None
  final: Metrics = None

  def to_dict(self):
      return {key: val.__dict__ for key, val in self.__dict__.items()}

def lp_norm(arr: Array, p: int = 2, axis: Sequence[int] = None) -> Array:
    """
    Returns the Bochner Lp-norm of an array.

    Args:
        arr: Point-wise values on a uniform grid with the dimensions
            [batch, time, space_0, space_1, var]
        p: Order of the norm. Defaults to 2.
        axis: The axes for to sum over. Defaults to None.

    Returns:
        A scalar value for each sample in the batch [batch, *remaining_axes]
    """

    # Set the axis
    if axis is None:
        axis = (1, 2, 3, 4)

    # Sum on timespace (quadrature) and variables
    abs_pow_sum = jnp.sum(jnp.power(jnp.abs(arr), p), axis=axis)
    # Take the p-th root
    pth_root = jnp.power(abs_pow_sum, (1/p))

    return pth_root

def rel_lp_error(gtr: Array, prd: Array, p: int = 2, vars: Sequence[int] = None) -> Array:
    """
    Returns the relative Bochner Lp-norm of an array with respect to a ground-truth.

    Args:
        gtr: Point-wise values of a ground-truth function on a uniform
            grid with the dimensions [batch, time, space_0, space_1, var]
        prd: Point-wise values of a predicted function on a uniform
            grid with the dimensions [batch, time, space_0, space_1, var]
        p: Order of the norm. Defaults to 2.
        vars: Index of the variables to use for computing the error. Defaults to None.

    Returns:
        A scalar value for each sample in the batch [batch,]
    """

    err = (prd - gtr)
    if vars is not None:
        err = err[..., vars]
    err_norm = lp_norm(err, p=p)
    gtr_norm = lp_norm(gtr, p=p)

    return (err_norm / (gtr_norm + EPSILON))

def rel_lp_error_per_var(gtr: Array, prd: Array, p: int = 2, vars: Sequence[int] = None) -> Array:
    """
    Returns the relative Bochner Lp-norm of an array with respect to a ground-truth.
    The entries of the last axis are interpreted as values of independent scalar-valued
    functions.

    Args:
        gtr: Point-wise values of a ground-truth function on a uniform
            grid with the dimensions [batch, time, space_0, space_1, var]
        prd: Point-wise values of a predicted function on a uniform
            grid with the dimensions [batch, time, space_0, space_1, var]
        p: Order of the norm. Defaults to 2.
        vars: Index of the variables to use for computing the error. Defaults to None.

    Returns:
        A scalar value for each sample in the batch [batch, var]
    """

    err = (prd - gtr)
    if vars is not None:
        err = err[..., vars]
    err_norm = lp_norm(err, p=p, axis=(1, 2, 3))
    gtr_norm = lp_norm(gtr, p=p, axis=(1, 2, 3))

    return (err_norm / (gtr_norm + EPSILON))

def rel_lp_error_norm(gtr: Array, prd: Array, p: int = 2, vars: Sequence[int] = None) -> Array:
    """
    Returns the norm of the relative Bochner Lp-norm of an array with respect to a ground-truth.
    The entries of the last axis are interpreted as values of independent scalar-valued
    functions. This results in an error vector. The vector norm of the error vector is returned.

    Args:
        gtr: Point-wise values of a ground-truth function on a uniform
            grid with the dimensions [batch, time, space_0, space_1, var]
        prd: Point-wise values of a predicted function on a uniform
            grid with the dimensions [batch, time, space_0, space_1, var]
        p: Order of the norm. Defaults to 2.
        vars: Index of the variables to use for computing the error. Defaults to None.

    Returns:
        The vector norm of the error vector [batch,]
    """

    err_per_var = rel_lp_error_per_var(gtr, prd, p=p, vars=vars)
    err_agg = jnp.linalg.norm(err_per_var, ord=p, axis=1)
    return err_agg

def rel_lp_loss(gtr: Array, prd: Array, p: int = 2) -> Array:
    """
    Returns the mean relative Bochner Lp-norm of an array with respect to a ground-truth.

    Args:
        gtr: Point-wise values of a ground-truth function on a uniform
            grid with the dimensions [batch, time, space_0, space_1, var]
        prd: Point-wise values of a predicted function on a uniform
            grid with the dimensions [batch, time, space_0, space_1, var]
        p: Order of the norm. Defaults to 2.

    Returns:
        Mean relative Lp-norm over the batch.
    """

    return jnp.mean(rel_lp_error_norm(gtr, prd, p=p))

def mse_error(gtr: Array, prd: Array) -> Array:
    """
    Returns the mean squared error per variable.
    All input shapes are [batch, time, space_0, space_1, var]
    Output shape is [batch,].
    """

    return jnp.mean(jnp.power(prd - gtr, 2), axis=(1, 2, 3, 4))

def mse_loss(gtr: Array, prd: Array) -> ScalarArray:
    """
    Returns the mean squared error.
    All input shapes are [batch, time, space_0, space_1, var]
    Output shape is a scalar.
    """

    return jnp.mean(jnp.power(prd - gtr, 2))
