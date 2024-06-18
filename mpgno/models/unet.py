from typing import Union

import jax
from jax import numpy as jnp
import flax.typing
from flax import linen as nn

from mpgno.models.mpgno import AbstractOperator
from mpgno.models.utils import ConditionedNorm
from mpgno.utils import Array




class Encoder(nn.Module):
  features: int
  conditional_norm_latent_size: int

  @nn.compact
  def __call__(self, x, tau):
    z1 = nn.Conv(self.features * 2, kernel_size=(3, 3))(x)
    z1 = nn.swish(z1)
    z1 = nn.Conv(self.features * 2, kernel_size=(7, 7))(z1)
    z1 = nn.LayerNorm()(z1)
    z1 = ConditionedNorm(
      latent_size=self.conditional_norm_latent_size,
      correction_size=(self.features * 2),
      convolutional=True,
    )(tau, z1)
    z1 = nn.swish(z1)
    z1_pool = nn.avg_pool(z1, window_shape=(2, 2), strides=(2, 2))

    z2 = nn.Conv(self.features * 4, kernel_size=(3, 3))(z1_pool)
    z2 = nn.swish(z2)
    z2 = nn.Conv(self.features * 4, kernel_size=(5, 5))(z2)
    z2 = nn.LayerNorm()(z2)
    z2 = ConditionedNorm(
      latent_size=self.conditional_norm_latent_size,
      correction_size=(self.features * 4),
      convolutional=True,
    )(tau, z2)
    z2 = nn.swish(z2)
    z2_pool = nn.avg_pool(z2, window_shape=(2, 2), strides=(2, 2))

    z3 = nn.Conv(self.features * 8, kernel_size=(3, 3))(z2_pool)
    z3 = nn.swish(z3)
    z3 = nn.Conv(self.features * 8, kernel_size=(3, 3))(z3)
    z3 = nn.LayerNorm()(z3)
    z3 = ConditionedNorm(
      latent_size=self.conditional_norm_latent_size,
      correction_size=(self.features * 8),
      convolutional=True,
    )(tau, z3)
    z3 = nn.swish(z3)
    # z3_dropout = nn.Dropout(0.5, deterministic=False)(z3)
    z3_dropout = z3
    z3_pool = nn.avg_pool(z3_dropout, window_shape=(2, 2), strides=(2, 2))

    z4 = nn.Conv(self.features * 16, kernel_size=(3, 3))(z3_pool)
    z4 = nn.swish(z4)
    z4 = nn.Conv(self.features * 16, kernel_size=(3, 3))(z4)
    z4 = nn.LayerNorm()(z4)
    z4 = ConditionedNorm(
      latent_size=self.conditional_norm_latent_size,
      correction_size=(self.features * 16),
      convolutional=True,
    )(tau, z4)
    z4 = nn.swish(z4)
    # z4_dropout = nn.Dropout(0.5, deterministic=False)(z4)
    z4_dropout = z4

    return z1, z2, z3_dropout, z4_dropout

class Decoder(nn.Module):
  features: int
  outputs: int
  conditional_norm_latent_size: int

  @nn.compact
  def __call__(self, z1, z2, z3_dropout, z4_dropout, tau):
    z5_up = jax.image.resize(
      z4_dropout,
      shape=(z4_dropout.shape[0], z4_dropout.shape[1] * 2, z4_dropout.shape[2] * 2, z4_dropout.shape[3]),
      method='nearest'
    )
    z5 = nn.Conv(self.features * 8, kernel_size=(2, 2))(z5_up)
    z5 = nn.swish(z5)
    z5 = jnp.concatenate([z3_dropout, z5], axis=3)
    z5 = nn.Conv(self.features * 8, kernel_size=(3, 3))(z5)
    z5 = nn.swish(z5)
    z5 = nn.Conv(self.features * 8, kernel_size=(3, 3))(z5)
    z5 = nn.LayerNorm()(z5)
    z5 = ConditionedNorm(
      latent_size=self.conditional_norm_latent_size,
      correction_size=(self.features * 8),
      convolutional=True,
    )(tau, z5)
    z5 = nn.swish(z5)

    z6_up = jax.image.resize(
      z5,
      shape=(z5.shape[0], z5.shape[1] * 2, z5.shape[2] * 2, z5.shape[3]),
      method='nearest'
    )
    z6 = nn.Conv(self.features * 4, kernel_size=(2, 2))(z6_up)
    z6 = nn.swish(z6)
    z6 = jnp.concatenate([z2, z6], axis=3)
    z6 = nn.Conv(self.features * 4, kernel_size=(5, 5))(z6)
    z6 = nn.swish(z6)
    z6 = nn.Conv(self.features * 4, kernel_size=(3, 3))(z6)
    z6 = nn.LayerNorm()(z6)
    z6 = ConditionedNorm(
      latent_size=self.conditional_norm_latent_size,
      correction_size=(self.features * 4),
      convolutional=True,
    )(tau, z6)
    z6 = nn.swish(z6)

    z7_up = jax.image.resize(
      z6,
      shape=(z6.shape[0], z6.shape[1] * 2, z6.shape[2] * 2, z6.shape[3]),
      method='nearest'
    )
    z7 = nn.Conv(self.features * 2, kernel_size=(2, 2))(z7_up)
    z7 = nn.swish(z7)
    z7 = jnp.concatenate([z1, z7], axis=3)
    z7 = nn.Conv(self.features * 2, kernel_size=(7, 7))(z7)
    z7 = nn.swish(z7)
    z7 = nn.Conv(self.features * 2, kernel_size=(3, 3))(z7)
    z7 = nn.LayerNorm()(z7)
    z7 = ConditionedNorm(
      latent_size=self.conditional_norm_latent_size,
      correction_size=(self.features * 2),
      convolutional=True,
    )(tau, z7)
    z7 = nn.swish(z7)

    y = nn.Conv(self.outputs, kernel_size=(1, 1))(z7)

    return y

class UNet(AbstractOperator):
  features: int
  outputs: int
  conditional_norm_latent_size: int
  concatenate_tau: bool = True
  concatenate_t: bool = True

  def setup(self):
    self.encoder = Encoder(
      features=self.features,
      conditional_norm_latent_size=self.conditional_norm_latent_size,
    )
    self.decoder = Decoder(
      features=self.features,
      outputs=self.outputs,
      conditional_norm_latent_size=self.conditional_norm_latent_size,
    )

  def __call__(self,
    u_inp: Array,
    t_inp: Array = None,
    tau: Union[float, int] = None,
    key: flax.typing.PRNGKey = None,
  ) -> Array:

    # Check the inputs
    batch_size = u_inp.shape[0]
    assert u_inp.shape[1] == 1
    assert u_inp.shape[-1] == self.outputs

    # Reshape tau and t_inp
    if self.concatenate_tau:
      assert tau is not None
      tau = jnp.array(tau, dtype=jnp.float32)
      if tau.size == 1:
        tau = jnp.tile(tau.reshape(1, 1), reps=(batch_size, 1))
    if self.concatenate_t:
      assert t_inp is not None
      t_inp = jnp.array(t_inp, dtype=jnp.float32)
      if t_inp.size == 1:
        t_inp = jnp.tile(t_inp.reshape(1, 1), reps=(batch_size, 1))

    # Concatenate tau and t_inp
    u_inp = u_inp.squeeze(axis=1)
    forced_features = []
    if self.concatenate_tau:
      forced_features.append(jnp.tile(tau.reshape(-1, 1, 1, 1), reps=(1, *u_inp.shape[1:3], 1)))
    if self.concatenate_t:
      forced_features.append(jnp.tile(t_inp.reshape(-1, 1, 1, 1), reps=(1, *u_inp.shape[1:3], 1)))
    input = jnp.concatenate([u_inp, *forced_features], axis=-1)

    z1, z2, z3_dropout, z4_dropout = self.encoder(input, tau)
    output = self.decoder(z1, z2, z3_dropout, z4_dropout, tau)
    output = jnp.expand_dims(output, axis=1)

    return output
