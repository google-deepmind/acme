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

"""A duelling network architecture, as described in [0].

[0] https://arxiv.org/abs/1511.06581
"""

from typing import Sequence

import haiku as hk
import jax.numpy as jnp


class DuellingMLP(hk.Module):
  """A Duelling MLP Q-network."""

  def __init__(
      self,
      num_actions: int,
      hidden_sizes: Sequence[int],
  ):
    super().__init__(name='duelling_q_network')

    self._value_mlp = hk.nets.MLP([*hidden_sizes, 1])
    self._advantage_mlp = hk.nets.MLP([*hidden_sizes, num_actions])

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Forward pass of the duelling network.

    Args:
      inputs: 2-D tensor of shape [batch_size, embedding_size].

    Returns:
      q_values: 2-D tensor of action values of shape [batch_size, num_actions]
    """

    # Compute value & advantage for duelling.
    value = self._value_mlp(inputs)  # [B, 1]
    advantages = self._advantage_mlp(inputs)  # [B, A]

    # Advantages have zero mean.
    advantages -= jnp.mean(advantages, axis=-1, keepdims=True)  # [B, A]

    q_values = value + advantages  # [B, A]

    return q_values
