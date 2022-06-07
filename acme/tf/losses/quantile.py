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

"""Losses for quantile regression."""

from typing import NamedTuple

from .huber import huber
import sonnet as snt
import tensorflow as tf


class QuantileDistribution(NamedTuple):
  values: tf.Tensor
  logits: tf.Tensor


class NonUniformQuantileRegression(snt.Module):
  """Compute the quantile regression loss for the distributional TD error."""

  def __init__(
      self,
      huber_param: float = 0.,
      name: str = 'NUQuantileRegression'):
    """Initializes the module.

    Args:
      huber_param: The point where the huber loss function changes from a
        quadratic to linear.
      name: name to use for grouping operations.
    """
    super().__init__(name=name)
    self._huber_param = huber_param

  def __call__(
      self,
      q_tm1: QuantileDistribution,
      r_t: tf.Tensor,
      pcont_t: tf.Tensor,
      q_t: QuantileDistribution,
      tau: tf.Tensor,
  ) -> tf.Tensor:
    """Calculates the loss.

    Note that this is only defined for discrete quantile-valued distributions.
    In particular we assume that the distributions define q.logits and
    q.values.

    Args:
      q_tm1: the distribution at time t-1.
      r_t: the reward at time t.
      pcont_t: the discount factor at time t.
      q_t: the target distribution.
      tau: the quantile regression targets.

    Returns:
      Value of the loss.
    """
    # Distributional Bellman update
    values_t = (tf.reshape(r_t, (-1, 1)) +
                tf.reshape(pcont_t, (-1, 1)) * q_t.values)
    values_t = tf.stop_gradient(values_t)
    probs_t = tf.nn.softmax(q_t.logits)

    # Quantile regression loss
    # Tau gives the quantile regression targets, where in the sample
    # space [0, 1] each output should train towards
    # Tau applies along the second dimension in delta (below)
    tau = tf.expand_dims(tau, -1)

    # quantile td-error and assymmetric weighting
    delta = values_t[:, None, :] - q_tm1.values[:, :, None]
    delta_neg = tf.cast(delta < 0., dtype=tf.float32)
    # This stop_gradient is very important, do not remove
    weight = tf.stop_gradient(tf.abs(tau - delta_neg))

    # loss
    loss = huber(delta, self._huber_param) * weight
    loss = tf.reduce_sum(loss * probs_t[:, None, :], 2)

    # Have not been able to get quite as good performance with mean vs. sum
    loss = tf.reduce_sum(loss, -1)
    return loss
