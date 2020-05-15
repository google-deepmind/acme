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

"""Modules for computing custom embeddings."""

from acme.networks import base
from acme.wrappers import observation_action_reward

import sonnet as snt
import tensorflow as tf


class OAREmbedding(snt.Module):
  """Module for embedding (observation, action, reward) inputs together."""

  def __init__(self, torso: base.Module, num_actions: int):
    super().__init__(name='oar_embedding')
    self._num_actions = num_actions
    self._torso = torso

  def __call__(self, inputs: observation_action_reward.OAR) -> tf.Tensor:
    """Embed each of the (observation, action, reward) inputs & concatenate."""

    # Add dummy trailing dimension to rewards if necessary.
    if len(inputs.reward.shape.dims) == 1:
      inputs = inputs._replace(reward=tf.expand_dims(inputs.reward, axis=-1))

    features = self._torso(inputs.observation)  # [T?, B, D]
    action = tf.one_hot(inputs.action, depth=self._num_actions)  # [T?, B, A]
    reward = tf.nn.tanh(inputs.reward)  # [T?, B, 1]

    embedding = tf.concat([features, action, reward], axis=-1)  # [T?, B, D+A+1]

    return embedding
