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

"""Policy-value network head for actor-critic algorithms."""

from typing import Tuple

import haiku as hk
import jax.numpy as jnp


class PolicyValueHead(hk.Module):
  """A network with two linear layers, for policy and value respectively."""

  def __init__(self, num_actions: int):
    super().__init__(name='policy_value_network')
    self._policy_layer = hk.Linear(num_actions)
    self._value_layer = hk.Linear(1)

  def __call__(self, inputs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns a (Logits, Value) tuple."""
    logits = self._policy_layer(inputs)  # [B, A]
    value = jnp.squeeze(self._value_layer(inputs), axis=-1)  # [B]

    return logits, value
