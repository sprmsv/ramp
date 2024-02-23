import jax.numpy as jnp


def error_rel_l2(predictions, labels):
    # shapes are [batch_size, num_times_output, num_grid_points, num_outputs]
    sqerr_per_var = jnp.mean(jnp.power(predictions - labels, 2), axis=(1, 2))
    rel_l2_err_per_var = jnp.sqrt(sqerr_per_var / jnp.mean(jnp.power(labels, 2), axis=(1, 2)))
    mean_rel_l2_err_per_var = jnp.mean(rel_l2_err_per_var, axis=0)

    return mean_rel_l2_err_per_var  # [num_outputs]

def loss_mse(predictions, labels):
    # shapes are [batch_size, num_times_output, num_grid_points, num_outputs]
    return jnp.mean(jnp.power(predictions - labels, 2))
