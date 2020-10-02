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

"""Networks used in discrete-action agents."""

import sonnet as snt
import tensorflow as tf


class DiscreteFilteredQNetwork(snt.Module):
  """Discrete filtered Q-network.

  This produces filtered Q values according to the method used in the discrete
  BCQ algorithm (https://arxiv.org/pdf/1910.01708.pdf - section 4).
  """

  def __init__(self,
               g_network: snt.Module,
               q_network: snt.Module,
               threshold: float):
    super().__init__(name='discrete_filtered_qnet')
    assert threshold >= 0 and threshold <= 1
    self.g_network = g_network
    self.q_network = q_network
    self._threshold = threshold

  def __call__(self, o_t: tf.Tensor) -> tf.Tensor:
    q_t = self.q_network(o_t)
    g_t = tf.nn.softmax(self.g_network(o_t))
    normalized_g_t = g_t / tf.reduce_max(g_t, axis=-1, keepdims=True)

    # Filter actions based on g_network outputs.
    min_q = tf.reduce_min(q_t, axis=-1, keepdims=True)
    return tf.where(normalized_g_t >= self._threshold, q_t, min_q)
