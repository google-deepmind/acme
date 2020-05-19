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

import functools
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


class ResidualBlock(hk.Module):
  """Residual block."""

  def __init__(self, num_channels: int, name: str = 'residual_block'):
    super().__init__(name=name)
    self._block = hk.Sequential([
        jax.nn.relu,
        hk.Conv2D(
            num_channels,
            kernel_shape=[3, 3],
            stride=[1, 1],
            padding='SAME'),
        jax.nn.relu,
        hk.Conv2D(
            num_channels,
            kernel_shape=[3, 3],
            stride=[1, 1],
            padding='SAME'),
    ])

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return self._block(x) + x


class DeepAtariTorso(hk.Module):
  """Deep torso for Atari, from the IMPALA paper."""

  def __init__(self, name: str = 'deep_atari_torso'):
    super().__init__(name=name)
    layers = []
    for i, (num_channels, num_blocks) in enumerate([(16, 2), (32, 2), (32, 2)]):
      conv = hk.Conv2D(
          num_channels, kernel_shape=[3, 3], stride=[1, 1], padding='SAME')
      pooling = functools.partial(
          hk.max_pool,
          window_shape=[1, 3, 3, 1],
          strides=[1, 2, 2, 1],
          padding='SAME')
      layers.append(conv)
      layers.append(pooling)

      for j in range(num_blocks):
        block = ResidualBlock(num_channels, name='residual_{}_{}'.format(i, j))
        layers.append(block)

    layers.extend([
        jax.nn.relu,
        hk.Flatten(),
        hk.Linear(256),
        jax.nn.relu,
    ])
    self._network = hk.Sequential(layers)

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return self._network(x)


class DeepIMPALAAtariNetwork(hk.RNNCore):
  """A recurrent network for use with IMPALA.

  See https://arxiv.org/pdf/1802.01561.pdf for more information.
  """

  def __init__(self, num_actions: int):
    super().__init__(name='impala_atari_network')
    self._torso = DeepAtariTorso()
    self._core = hk.LSTM(256)
    self._head = policy_value.PolicyValueHead(num_actions)
    self._num_actions = num_actions

  def __call__(self, inputs: observation_action_reward.OAR,
               state: LSTMState) -> Tuple[Tuple[Logits, Value], LSTMState]:
    reward = jnp.tanh(inputs.reward)  # [B, 1]
    if not reward.shape:
      reward = jnp.expand_dims(reward, axis=0)

    action = hk.one_hot(inputs.action, self._num_actions)  # [B, A]

    expand = len(inputs.observation.shape) == 3
    if expand:
      inputs = inputs._replace(
          observation=jnp.expand_dims(inputs.observation, axis=0))
    embedding = self._torso(inputs.observation)
    if expand:
      embedding = jnp.squeeze(embedding, axis=0)
    embedding = jnp.concatenate([embedding, action, reward], axis=-1)

    embedding, new_state = self._core(embedding, state)
    logits, value = self._head(embedding)  # [B, A]

    return (logits, value), new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> LSTMState:
    return self._core.initial_state(batch_size)
