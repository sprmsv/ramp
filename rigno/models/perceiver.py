from functools import partial
import jax.nn.initializers as init
import jax.numpy as jnp
from einops import rearrange, repeat
from flax import linen as nn


def default(val, d):
    return val if val is not None else d


def fourier_encode(x: jnp.ndarray, num_encodings=4):
    x = jnp.expand_dims(x, -1)
    orig_x = x
    scales = 2 ** jnp.arange(num_encodings)
    x /= scales
    x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
    x = jnp.concatenate([x, orig_x], axis=-1)
    return x


class FeedForward(nn.Module):
    mult: int = 4
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic=False):
        features = x.shape[-1]
        x = nn.Dense(features * self.mult)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout)(x, deterministic=deterministic)
        x = nn.Dense(features)(x)
        return x


class Attention(nn.Module):
    heads: int = 8
    head_features: int = 64
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, context=None, mask=None, deterministic=False):
        h = self.heads
        dim = self.head_features * h

        q = nn.Dense(dim, use_bias=False)(x)
        k, v = jnp.split(nn.Dense(dim * 2, use_bias=False)(default(context, x)), 2, axis=-1)

        q, k, v = map(
            lambda arr: rearrange(arr, "b n (h d) -> (b h) n d", h=h), (q, k, v)
        )
        sim = jnp.einsum("b i d, b j d -> b i j", q, k) * self.head_features ** -0.5
        attn = nn.softmax(sim, axis=-1)
        out = jnp.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        out = nn.Dense(x.shape[-1])(out)
        out = nn.Dropout(self.dropout)(out, deterministic=deterministic)
        return out


class BCProjector(nn.Module):
  depth: int = 2
  out_dim: int = 4
  n_heads: int = 2
  head_dim: int = 128
  ff_mult: int = 4
  attn_dropout: float = 0.0
  ff_dropout: float = 0.0

  @nn.compact
  def __call__(self, f_boundary, f_domain, train: bool = False):
    # TODO: Use the 'train' argument for all dropouts
    f_boundary = rearrange(f_boundary, "b n ... -> b n (...)")
    # x_domain = fourier_encode(x_domain, self.n_fourier_features)  # TRY: Enable this line

    f_domain = nn.Dense(features=self.ff_mult*self.out_dim)(f_domain)
    f_domain = nn.gelu(f_domain)
    f_domain = nn.Dropout(self.ff_dropout)(f_domain, deterministic=(not train))
    f_domain = nn.Dense(features=self.out_dim)(f_domain)
    f_domain = nn.LayerNorm()(f_domain)

    cross_attn = partial(
        Attention,
        heads=self.n_heads,
        head_features=self.head_dim,
        dropout=self.attn_dropout,
    )
    ff = partial(FeedForward, mult=self.ff_mult, dropout=self.ff_dropout)

    for i in range(self.depth):
      f_domain += cross_attn(name=f"cross_attn_{i}")(f_domain, f_boundary)
      f_domain = nn.LayerNorm()(f_domain)
      f_domain += ff(name=f"cross_ff_{i}")(f_domain)
      f_domain = nn.LayerNorm()(f_domain)

    return f_domain
