import jax.numpy as jnp
from graphneuralpdesolver.utils import Array, ScalarArray


def rel_l2_error(predictions: Array, labels: Array) -> Array:
    """
    Returns the relative L2-norm of the error per variable.
    Input shapes are [batch_size, num_times_output, num_grid_points, num_outputs]
    Output shape is [num_outputs].
    """

    sum_err_per_var_squared = jnp.sum(jnp.power(predictions - labels, 2), axis=(1, 2))
    sum_lab_per_var_squared = jnp.sum(jnp.power(labels, 2), axis=(1, 2))
    rel_l2_err_per_var = jnp.sqrt(sum_err_per_var_squared / sum_lab_per_var_squared)
    mean_rel_l2_err_per_var = jnp.mean(rel_l2_err_per_var, axis=0)

    return mean_rel_l2_err_per_var

def rel_l1_error(predictions: Array, labels: Array) -> Array:
    """
    Returns the relative L1-norm of the error per variable.
    Input shapes are [batch_size, num_times_output, num_grid_points, num_outputs].
    Output shape is [num_outputs].
    """

    sum_err_per_var_abs = jnp.sum(predictions - labels, axis=(1, 2))
    sum_lab_per_var_abs = jnp.sum(labels, axis=(1, 2))
    rel_l1_err_per_var = (sum_err_per_var_abs / sum_lab_per_var_abs)
    mean_rel_l2_err_per_var = jnp.mean(rel_l1_err_per_var, axis=0)

    return mean_rel_l2_err_per_var

def mse(predictions: Array, labels: Array) -> ScalarArray:
    """
    Returns the mean squared error.
    Input shapes are [batch_size, num_times_output, num_grid_points, num_outputs].
    Output shape is [1].
    """

    return jnp.mean(jnp.power(predictions - labels, 2))
