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

"""Commonly-used networks for running on Atari."""

from typing import Tuple

from acme.networks import base
from acme.networks import duelling
from acme.networks import embedding
from acme.networks import policy_value
from acme.networks import vision
from acme.wrappers import observation_action_reward

import sonnet as snt
import tensorflow as tf

Images = tf.Tensor
QValues = tf.Tensor
Logits = tf.Tensor
Value = tf.Tensor


class AtariTorso(base.Module):
  """Simple convolutional stack commonly used for Atari."""

  def __init__(self):
    super().__init__(name='atari_torso')
    self._network = snt.Sequential([
        snt.Conv2D(32, [8, 8], [4, 4]),
        tf.nn.relu,
        snt.Conv2D(64, [4, 4], [2, 2]),
        tf.nn.relu,
        snt.Conv2D(64, [3, 3], [1, 1]),
        tf.nn.relu,
        snt.Flatten(),
    ])

  def __call__(self, inputs: Images) -> tf.Tensor:
    return self._network(inputs)


class DQNAtariNetwork(base.Module):
  """A feed-forward network for use with Ape-X DQN.

  See https://arxiv.org/pdf/1803.00933.pdf for more information.
  """

  def __init__(self, num_actions: int):
    super().__init__(name='dqn_atari_network')
    self._network = snt.Sequential([
        AtariTorso(),
        duelling.DuellingMLP(num_actions, hidden_sizes=[512]),
    ])

  def __call__(self, inputs: Images) -> QValues:
    return self._network(inputs)


class R2D2AtariNetwork(base.RNNCore):
  """A recurrent network for use with R2D2.

  See https://openreview.net/forum?id=r1lyTjAqYX for more information.
  """

  def __init__(self, num_actions: int):
    super().__init__(name='r2d2_atari_network')
    self._embed = embedding.OAREmbedding(
        torso=AtariTorso(), num_actions=num_actions)
    self._core = snt.LSTM(512)
    self._head = duelling.DuellingMLP(num_actions, hidden_sizes=[512])

  def __call__(
      self,
      inputs: observation_action_reward.OAR,
      state: snt.LSTMState,
  ) -> Tuple[QValues, snt.LSTMState]:

    embeddings = self._embed(inputs)
    embeddings, new_state = self._core(embeddings, state)
    action_values = self._head(embeddings)  # [B, A]

    return action_values, new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> snt.LSTMState:
    return self._core.initial_state(batch_size)

  def unroll(
      self,
      inputs: observation_action_reward.OAR,
      state: snt.LSTMState,
      sequence_length: int,
  ) -> Tuple[QValues, snt.LSTMState]:
    """Efficient unroll that applies embeddings, MLP, & convnet in one pass."""
    embeddings = snt.BatchApply(self._embed)(inputs)  # [T, B, D+A+1]
    embeddings, new_state = snt.static_unroll(self._core, embeddings, state,
                                              sequence_length)
    action_values = snt.BatchApply(self._head)(embeddings)

    return action_values, new_state


class IMPALAAtariNetwork(snt.RNNCore):
  """A recurrent network for use with IMPALA.

  See https://arxiv.org/pdf/1802.01561.pdf for more information.
  """

  def __init__(self, num_actions: int):
    super().__init__(name='impala_atari_network')
    self._embed = embedding.OAREmbedding(
        torso=AtariTorso(), num_actions=num_actions)
    self._core = snt.LSTM(256)
    self._head = snt.Sequential([
        snt.Linear(256),
        tf.nn.relu,
        policy_value.PolicyValueHead(num_actions),
    ])
    self._num_actions = num_actions

  def __call__(
      self, inputs: observation_action_reward.OAR,
      state: snt.LSTMState) -> Tuple[Tuple[Logits, Value], snt.LSTMState]:

    embeddings = self._embed(inputs)
    embeddings, new_state = self._core(embeddings, state)
    logits, value = self._head(embeddings)  # [B, A]

    return (logits, value), new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> snt.LSTMState:
    return self._core.initial_state(batch_size)


class DeepIMPALAAtariNetwork(base.RNNCore):
  """A recurrent network for use with IMPALA.

  See https://arxiv.org/pdf/1802.01561.pdf for more information.
  """

  def __init__(self, num_actions: int):
    super().__init__(name='deep_impala_atari_network')
    self._embed = embedding.OAREmbedding(
        torso=vision.ResNetTorso(), num_actions=num_actions)
    self._core = snt.LSTM(256)
    self._head = snt.Sequential([
        snt.Linear(256),
        tf.nn.relu,
        policy_value.PolicyValueHead(num_actions),
    ])
    self._num_actions = num_actions

  def __call__(
      self, inputs: observation_action_reward.OAR,
      state: snt.LSTMState) -> Tuple[Tuple[Logits, Value], snt.LSTMState]:

    embeddings = self._embed(inputs)
    embeddings, new_state = self._core(embeddings, state)
    logits, value = self._head(embeddings)  # [B, A]

    return (logits, value), new_state

  def initial_state(self, batch_size: int, **unused_kwargs) -> snt.LSTMState:
    return self._core.initial_state(batch_size)

  def unroll(
      self,
      inputs: observation_action_reward.OAR,
      states: snt.LSTMState,
      sequence_length: int,
  ) -> Tuple[Tuple[Logits, Value], snt.LSTMState]:
    """Efficient unroll that applies embeddings, MLP, & convnet in one pass."""
    embeddings = snt.BatchApply(self._embed)(inputs)  # [T, B, D+A+1]
    embeddings, new_states = snt.static_unroll(self._core, embeddings, states,
                                               sequence_length)
    logits, values = snt.BatchApply(self._head)(embeddings)

    return (logits, values), new_states
