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

"""An implicit quantile network, as described in [0].

[0] https://arxiv.org/abs/1806.06923
"""

import numpy as np
import sonnet as snt
import tensorflow as tf


class IQNNetwork(snt.Module):
  """A feedforward network for use with IQN.

  IQN extends the Q-network of regular DQN which consists of torso and head
  networks. IQN embeds sampled quantile thresholds into the output space of the
  torso network and merges them with the torso output.

  Outputs a tuple consisting of (mean) Q-values, Q-value quantiles, and sampled
  quantile thresholds.
  """

  def __init__(self,
               torso: snt.Module,
               head: snt.Module,
               latent_dim: int,
               num_quantile_samples: int,
               name: str = 'iqn_network'):
    """Initializes the network.

    Args:
      torso: Network producing an intermediate representation, typically a
        convolutional network.
      head: Network producing Q-value quantiles, typically an MLP.
      latent_dim: Dimension of latent variables.
      num_quantile_samples: Number of quantile thresholds to sample.
      name: Module name.
    """
    super().__init__(name)
    self._torso = torso
    self._head = head
    self._latent_dim = latent_dim
    self._num_quantile_samples = num_quantile_samples

  @snt.once
  def _create_embedding(self, size):
    self._embedding = snt.Linear(size)

  def __call__(self, observations):
    # Transform observations to intermediate representations (typically a
    # convolutional network).
    torso_output = self._torso(observations)

    # Now that dimension of intermediate representation is known initialize
    # embedding of sample quantile thresholds (only done once).
    self._create_embedding(torso_output.shape[-1])

    # Sample quantile thresholds.
    batch_size = tf.shape(observations)[0]
    tau_shape = tf.stack([batch_size, self._num_quantile_samples])
    tau = tf.random.uniform(tau_shape)
    indices = tf.range(1, self._latent_dim+1, dtype=tf.float32)

    # Embed sampled quantile thresholds in intermediate representation space.
    tau_tiled = tf.tile(tau[:, :, None], (1, 1, self._latent_dim))
    indices_tiled = tf.tile(indices[None, None, :],
                            tf.concat([tau_shape, [1]], 0))
    tau_embedding = tf.cos(tau_tiled * indices_tiled * np.pi)
    tau_embedding = snt.BatchApply(self._embedding)(tau_embedding)
    tau_embedding = tf.nn.relu(tau_embedding)

    # Merge intermediate representations with embeddings, and apply head
    # network (typically an MLP).
    torso_output = tf.tile(torso_output[:, None, :],
                           (1, self._num_quantile_samples, 1))
    q_value_quantiles = snt.BatchApply(self._head)(tau_embedding * torso_output)
    q_dist = tf.transpose(q_value_quantiles, (0, 2, 1))
    q_values = tf.reduce_mean(q_value_quantiles, axis=1)
    q_values = tf.stop_gradient(q_values)

    return q_values, q_dist, tau
