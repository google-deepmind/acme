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

from acme.jax.networks import base
from acme.wrappers import observation_action_reward

import dataclasses
import haiku as hk
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class OAREmbedding(hk.Module):
  """Module for embedding (observation, action, reward) inputs together."""

  torso: base.Module
  num_actions: int

  def __call__(self, inputs: observation_action_reward.OAR) -> jnp.ndarray:
    """Embed each of the (observation, action, reward) inputs & concatenate."""

    # Add dummy batch dimension to observation if necessary.
    # This is needed because Conv2D assumes a leading batch dimension, i.e.
    # that inputs are in [B, H, W, C] format.
    expand_obs = len(inputs.observation.shape) == 3
    if expand_obs:
      inputs = inputs._replace(
          observation=jnp.expand_dims(inputs.observation, axis=0))
    features = self.torso(inputs.observation)  # [T?, B, D]
    if expand_obs:
      features = jnp.squeeze(features, axis=0)

    # Do a one-hot embedding of the actions.
    action = jax.nn.one_hot(
        inputs.action, num_classes=self.num_actions)  # [T?, B, A]

    # Map rewards -> [-1, 1].
    reward = jnp.tanh(inputs.reward)

    # Add dummy trailing dimensions to rewards if necessary.
    while reward.ndim < action.ndim:
      reward = jnp.expand_dims(reward, axis=-1)

    # Concatenate on final dimension.
    embedding = jnp.concatenate(
        [features, action, reward], axis=-1)  # [T?, B, D+A+1]

    return embedding
