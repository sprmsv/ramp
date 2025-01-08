from typing import Union, NamedTuple

from flax import linen as nn

from rigno.utils import Array


class Inputs(NamedTuple):
  u: Array  # Contains the full-domain functions
  h: Array  # Contains the segment function values, accomponied by a mask
  m: Array  # Binary function masks, make sure to not normalize at all
  x_inp: Array
  x_out: Array
  t: Union[Array, float, None]
  tau: Union[Array, float, None]

class AbstractOperator(nn.Module):
  def setup(self):
    raise NotImplementedError

  def __call__(self,
    inputs: Inputs,
    **kwargs,
  ) -> Array:
    return self.call(inputs, **kwargs)

  def call(self, inputs: Inputs) -> Array:
    raise NotImplementedError

  @property
  def configs(self):
    configs = {
      attr: self.__getattr__(attr)
      for attr in self.__annotations__.keys() if attr != 'parent'
    }
    return configs
