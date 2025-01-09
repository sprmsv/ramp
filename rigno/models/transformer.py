import flax.linen as nn


class EncoderBlock(nn.Module):
  inp_dim: int
  num_heads: int
  qkv_dim: int
  ff_dim: int
  dropout_prob: float

  def setup(self):
    # Attention layer
    self.self_attn = nn.MultiHeadAttention(
      num_heads=self.num_heads,
      qkv_features=self.qkv_dim,
      out_features=self.inp_dim,
      kernel_init=nn.initializers.xavier_uniform(),
      decode=False,
      dropout_rate=self.dropout_prob,
      deterministic=None,
    )
    # Two-layer MLP
    self.linear = [
      nn.Dense(self.ff_dim),
      nn.Dropout(self.dropout_prob),
      nn.swish,
      nn.Dense(self.inp_dim)
    ]
    # Layers to apply in between the main layers
    self.norm1 = nn.LayerNorm()
    self.norm2 = nn.LayerNorm()
    self.dropout = nn.Dropout(self.dropout_prob)

  def __call__(self, x, train=True):
    # Attention part
    attn_out = self.self_attn(x, deterministic=(not train))
    x = x + self.dropout(attn_out, deterministic=(not train))
    x = self.norm1(x)

    # MLP part
    linear_out = x
    for l in self.linear:
      linear_out = l(linear_out) if not isinstance(l, nn.Dropout) else l(linear_out, deterministic=(not train))
    x = x + self.dropout(linear_out, deterministic=(not train))
    x = self.norm2(x)

    return x

class TransformerEncoder(nn.Module):
  inp_dim: int
  num_layers: int
  num_heads: int
  qkv_dim: int
  ff_dim: int
  dropout_prob: float

  def setup(self):
    self.layers = [
      EncoderBlock(
        num_heads=self.num_heads,
        qkv_dim=self.qkv_dim,
        inp_dim=self.inp_dim,
        ff_dim=self.ff_dim,
        dropout_prob=self.dropout_prob,
      )
      for _ in range(self.num_layers)
    ]

  def __call__(self, x, train=True):
    for l in self.layers:
      x = l(x, train=train)
    return x
