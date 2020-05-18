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

"""Common networks for Atari."""

from typing import Any, Tuple

from acme.networks.jax import base
from acme.networks.jax import duelling
from acme.networks.jax import policy_value

from acme.wrappers import observation_action_reward

import haiku as hk
import jax
import jax.numpy as jnp

# Useful type aliases.
Images = jnp.ndarray
Logits = jnp.ndarray
Value = jnp.ndarray
LSTMState = Any


class AtariTorso(hk.Module):
  """Simple convolutional stack commonly used for Atari."""

  def __init__(self):
    super().__init__(name='atari_torso')
    self._network = hk.Sequential([
        hk.Conv2D(32, [8, 8], 4),
        jax.nn.relu,
        hk.Conv2D(64, [4, 4], 2),
        jax.nn.relu,
        hk.Conv2D(64, [3, 3], 1),
        jax.nn.relu,
        hk.Flatten(),
    ])

  def __call__(self, inputs: Images) -> jnp.ndarray:
    return self._network(inputs)


def dqn_atari_network(num_actions: int) -> base.QNetwork:
  """A feed-forward network for use with Ape-X DQN."""

  def network(inputs: Images) -> base.QValues:
    model = hk.Sequential([
        AtariTorso(),
        duelling.DuellingMLP(num_actions, hidden_sizes=[512]),
    ])
    return model(inputs)

  return network


class IMPALAAtariNetwork(hk.RNNCore):
  """A recurrent network for use with IMPALA.

  See https://arxiv.org/pdf/1802.01561.pdf for more information.
  """

  def __init__(self, num_actions: int):
    super().__init__(name='impala_atari_network')
    self._torso = AtariTorso()
    self._core = hk.LSTM(256)
    self._head = hk.Sequential([
        hk.Linear(512),
        jax.nn.relu,
        policy_value.PolicyValueHead(num_actions),
    ])
    self._num_actions = num_actions

  def __call__(
      self, inputs: observation_action_reward.OAR,
      state: LSTMState) -> Tuple[Tuple[Logits, Value], LSTMState]:
    if len(inputs.reward.shape.dims) == 1:
      inputs = inputs._replace(reward=jnp.expand_dims(inputs.reward, axis=-1))
    reward = jnp.tanh(inputs.reward)  # [B, 1]

    action = hk.one_hot(inputs.action, self._num_actions)  # [B, A]

    embedding = self._torso(inputs.observation)
    embedding = jnp.concatenate([embedding, action, reward], axis=-1)

    embedding, new_state = self._core(embedding, state)
    logits, value = self._head(embedding)  # [B, A]

    return (logits, value), new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> LSTMState:
    return self._core.initial_state(batch_size)
