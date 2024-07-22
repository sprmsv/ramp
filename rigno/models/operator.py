from typing import Union

import flax.typing
from flax import linen as nn

from rigno.utils import Array


class AbstractOperator(nn.Module):
  def setup(self):
    raise NotImplementedError

  def __call__(self,
    u_inp: Array,
    c_inp: Array = None,
    x_inp: Array = None,
    x_out: Array = None,
    t_inp: Array = None,
    tau: Union[float, int] = None,
    key: flax.typing.PRNGKey = None,
  ) -> Array:
    raise self.call(u_inp, c_inp, x_inp, x_out, t_inp, tau, key)

  def call(self,
    u_inp: Array,
    c_inp: Array,
    x_inp: Array,
    x_out: Array,
    t_inp: Array,
    tau: Union[float, int],
    key: flax.typing.PRNGKey,
  ) -> Array:
    raise NotImplementedError

  @property
  def configs(self):
    configs = {
      attr: self.__getattr__(attr)
      for attr in self.__annotations__.keys() if attr != 'parent'
    }
    return configs
