# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multihead networks apply separate networks to the input."""

from typing import Callable, Union, Sequence

from acme import types

import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
TensorTransformation = Union[snt.Module, Callable[[types.NestedTensor],
                                                  tf.Tensor]]


class Multihead(snt.Module):
  """Multi-head network module.

  This takes as input a list of N `network_heads`, and returns another network
  whose output is the stacked outputs of each of these network heads separately
  applied to the module input. The dimension of the output is [..., N].
  """

  def __init__(self,
               network_heads: Sequence[TensorTransformation]):
    if not network_heads:
      raise ValueError('Must specify non-empty, non-None critic_network_heads.')
    self._network_heads = network_heads
    super().__init__(name='multihead')

  def __call__(self,
               inputs: tf.Tensor) -> Union[tf.Tensor, Sequence[tf.Tensor]]:
    outputs = [network_head(inputs) for network_head in self._network_heads]
    if isinstance(outputs[0], tfd.Distribution):
      # Cannot stack distributions
      return outputs
    outputs = tf.stack(outputs, axis=-1)
    return outputs
