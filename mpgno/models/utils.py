"""A library of auxiliary functions and classes."""

from typing import Sequence, Callable

import jax.numpy as jnp
import jax.tree_util as tree
import flax.linen as nn

from mpgno.utils import Array, calculate_fd_derivative


def concatenate_args(args, kwargs, axis: int = -1):
  combined_args = tree.tree_flatten(args)[0] + tree.tree_flatten(kwargs)[0]
  concat_args = jnp.concatenate(combined_args, axis=axis)
  return concat_args

def compute_derivatives(traj: Array, degree: int = 1):
  """Returns spatial derivatives."""

  if degree < 1:
    return None

  grads = []
  if degree >= 1:
    g_x, g_y = calculate_fd_derivative(traj, axes=(2, 3))
    grads.extend([g_x, g_y])
  if degree >= 2:
    g_xx, g_xy = calculate_fd_derivative(g_x, axes=(2, 3))
    _, g_yy = calculate_fd_derivative(g_y, axes=(2, 3))
    grads.extend([g_xx, g_yy, g_xy])

  grads = jnp.concatenate(grads, axis=-1)

  return grads

class AugmentedMLP(nn.Module):
  """
  Multi-layer perceptron with optional layer norm and learned correction on the last layer.
  Activation is applied on all layers except the last one.
  Multiple inputs are concatenated before being fed to the MLP.
  """

  layer_sizes: Sequence[int]
  activation: Callable
  use_layer_norm: bool = False
  use_conditional_norm: bool = False
  conditional_norm_latent_size: int = 4
  conditional_norm_unique: bool = True
  conditional_norm_nonlinear: bool = True
  concatenate_axis: int = -1

  def setup(self):
    # Set up layers
    self.layers = [nn.Dense(features) for features in self.layer_sizes]

    # Set up normalization layer
    self.layernorm = nn.LayerNorm(
      reduction_axes=-1,
      feature_axes=-1,
      use_scale=True,
      use_bias=True,
    ) if self.use_layer_norm else None

    # Set conditional normalization layer
    self.correction = None
    if self.use_conditional_norm:
      if self.conditional_norm_nonlinear:
        self.correction = LearnedCorrection(
          latent_size=self.conditional_norm_latent_size,
          correction_size=(1 if self.conditional_norm_unique else self.layer_sizes[-1]),
        )
      else:
        self.correction = LinearLearnedCorrection(
          correction_size=(1 if self.conditional_norm_unique else self.layer_sizes[-1]),
        )

  def __call__(self, *args, c = None, **kwargs):
    x = concatenate_args(args=args, kwargs=kwargs, axis=self.concatenate_axis)
    for layer in self.layers[:-1]:
      x = layer(x)
      x = self.activation(x)
    x = self.layers[-1](x)
    if self.layernorm:
      x = self.layernorm(x)
    if self.correction:
      assert c is not None
      x = self.correction(c=c, x=x)
    return x

class LearnedCorrection(nn.Module):
  """
  Learned correction layer is designed to be applied after a normalization layer.
  Based on an input (e.g., time delta), it shifts and scales the distribution of its input.
  correction_size must either be 1 or the same as one of the input dimensions (broadcastable).
  """

  latent_size: Sequence[int]
  correction_size: int = 1

  def setup(self):
    self.mlp_scale = nn.Sequential(
      layers=[
        nn.Dense(self.latent_size,
                 kernel_init=nn.initializers.normal(stddev=.01)),
        nn.sigmoid,
        nn.Dense(self.correction_size,
                 kernel_init=nn.initializers.normal(stddev=.01)),
      ])
    self.mlp_bias = nn.Sequential(
      layers=[
        nn.Dense(self.latent_size,
                 kernel_init=nn.initializers.normal(stddev=.01)),
        nn.sigmoid,
        nn.Dense(self.correction_size,
                 kernel_init=nn.initializers.normal(stddev=.01)),
      ])

  def __call__(self, c, x):
    scale = 1 + c * self.mlp_scale(c)
    bias = c * self.mlp_bias(c)
    return x * scale + bias

class LinearLearnedCorrection(nn.Module):
  """
  Learned correction layer is designed to be applied after a normalization layer.
  Based on an input (e.g., time delta), it shifts and scales the distribution of its input.
  correction_size must either be 1 or the same as one of the input dimensions (broadcastable).
  """

  correction_size: int = 1

  def setup(self):
    self.mlp_scale = nn.Sequential(
      layers=[
        nn.Dense(self.correction_size,
                 kernel_init=nn.initializers.normal(stddev=.01),
                 bias_init=nn.initializers.constant(1.))
      ])
    self.mlp_bias = nn.Sequential(
      layers=[
        nn.Dense(self.correction_size,
                 kernel_init=nn.initializers.normal(stddev=.01),
                 bias_init=nn.initializers.constant(0.))
      ])

  def __call__(self, c, x):
    scale = self.mlp_scale(c)
    bias = self.mlp_bias(c)
    return x * scale + bias
