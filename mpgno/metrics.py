from dataclasses import dataclass
from copy import copy

import jax.numpy as jnp

from mpgno.utils import Array, ScalarArray


@dataclass
class BatchMetrics:
    mse: Array = None
    l1: Array = None
    l2: Array = None
    l1_alt: Array = None
    l2_alt: Array = None

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
    l1_alt: float = None
    l2_alt: float = None

@dataclass
class EvalMetrics:
  direct: Metrics = None
  rollout: Metrics = None
  final: Metrics = None

  def to_dict(self):
      return {key: val.__dict__ for key, val in self.__dict__.items()}

def mse_error(predictions: Array, labels: Array) -> Array:
    """
    Returns the mean squared error per variable.
    All input shapes are [batch_size, num_times_output,
        num_grid_points_0, num_grid_points_1, num_outputs]
    Output shape is [batch_size, num_outputs].
    """

    mean_err_per_var_squared = jnp.mean(jnp.power(predictions - labels, 2), axis=(1, 2, 3))

    return mean_err_per_var_squared

def rel_l1_error(predictions: Array, labels: Array) -> Array:
    """
    Returns the relative L1-norm of the error per variable.
    All input shapes are [batch_size, num_times_output,
        num_grid_points_0, num_grid_points_1, num_outputs]
    Output shape is [batch_size, num_outputs].
    """

    sum_err_per_var_abs = jnp.sum(jnp.abs(predictions - labels), axis=(1, 2, 3))
    sum_lab_per_var_abs = jnp.sum(jnp.abs(labels), axis=(1, 2, 3))
    rel_l1_err_per_var = (sum_err_per_var_abs / sum_lab_per_var_abs)

    return rel_l1_err_per_var

def rel_l2_error(predictions: Array, labels: Array) -> Array:
    """
    Returns the relative L2-norm of the error per variable.
    All input shapes are [batch_size, num_times_output,
        num_grid_points_0, num_grid_points_1, num_outputs]
    Output shape is [batch_size, num_outputs].
    """

    sum_err_per_var_squared = jnp.sum(jnp.power(predictions - labels, 2), axis=(1, 2, 3))
    sum_lab_per_var_squared = jnp.sum(jnp.power(labels, 2), axis=(1, 2, 3))
    rel_l2_err_per_var = jnp.sqrt(sum_err_per_var_squared / sum_lab_per_var_squared)

    return rel_l2_err_per_var

def rel_l1_error_sum_vars(predictions: Array, labels: Array) -> Array:
    """
    Returns the relative L1-norm of the error for all variables.
    All input shapes are [batch_size, num_times_output,
        num_grid_points_0, num_grid_points_1, num_outputs]
    Output shape is [batch_size,].
    """

    sum_err_abs = jnp.sum(jnp.abs(predictions - labels), axis=(1, 2, 3, 4))
    sum_lab_abs = jnp.sum(jnp.abs(labels), axis=(1, 2, 3, 4))
    rel_l1_err = (sum_err_abs / sum_lab_abs)

    return rel_l1_err

def rel_l2_error_sum_vars(predictions: Array, labels: Array) -> Array:
    """
    Returns the relative L2-norm of the error for all variables.
    All input shapes are [batch_size, num_times_output,
        num_grid_points_0, num_grid_points_1, num_outputs]
    Output shape is [batch_size,].
    """

    sum_err_squared = jnp.sum(jnp.power(predictions - labels, 2), axis=(1, 2, 3, 4))
    sum_lab_squared = jnp.sum(jnp.power(labels, 2), axis=(1, 2, 3, 4))
    rel_l2_err = jnp.sqrt(sum_err_squared / sum_lab_squared)

    return rel_l2_err

def mse_loss(predictions: Array, labels: Array) -> ScalarArray:
    """
    Returns the mean squared error.
    Input shapes are [batch_size, num_times_output, num_grid_points, num_outputs].
    Output shape is [1].
    """

    return jnp.mean(jnp.power(predictions - labels, 2))

def msre_loss(predictions: Array, labels: Array) -> ScalarArray:
    """
    Returns the mean squared relative error.
    Input shapes are [batch_size, num_times_output, num_grid_points, num_outputs].
    Output shape is [1].
    """

    eps = 1e-08
    return jnp.mean(jnp.power((predictions - labels) / (labels + eps), 2))

def rel_l1_loss(predictions: Array, labels: Array) -> ScalarArray:
    """
    Returns the mean relative L1-norm loss.
    Input shapes are [batch_size, num_times_output, num_grid_points, num_outputs].
    Output shape is [1].
    """

    rel_err_per_var = rel_l1_error(predictions, labels)
    rel_err_agg = jnp.linalg.norm(rel_err_per_var, axis=-1)

    return jnp.mean(rel_err_agg)

def rel_l2_loss(predictions: Array, labels: Array) -> ScalarArray:
    """
    Returns the mean relative L2-norm loss.
    Input shapes are [batch_size, num_times_output, num_grid_points, num_outputs].
    Output shape is [1].
    """

    rel_err_per_var = rel_l2_error(predictions, labels)
    rel_err_agg = jnp.linalg.norm(rel_err_per_var, axis=-1)

    return jnp.mean(rel_err_agg)
