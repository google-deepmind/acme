# Lint as: python3
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

"""Distributions, for use in acme/networks/distributional.py."""

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class DiscreteValuedDistribution(tfd.Categorical):
  """This is a generalization of a categorical distribution.

  The support for the DiscreteValued distribution can be any real valued range,
  whereas the categorical distribution has support [0, n_categories - 1] or
  [1, n_categories]. This generalization allows us to take the mean of the
  distribution over its support.
  """

  def __init__(self,
               values: tf.Tensor,
               logits: tf.Tensor = None,
               probs: tf.Tensor = None):
    """Initialization.

    Args:
      values: Values making up support of the distribution. Should have a shape
        compatible with logits.
      logits: An N-D Tensor, N >= 1, representing the log probabilities of a set
        of Categorical distributions. The first N - 1 dimensions index into a
        batch of independent distributions and the last dimension indexes into
        the classes.
      probs: An N-D Tensor, N >= 1, representing the probabilities of a set of
        Categorical distributions. The first N - 1 dimensions index into a batch
        of independent distributions and the last dimension represents a vector
        of probabilities for each class. Only one of logits or probs should be
        passed in.
    """
    super().__init__(
        logits=logits, probs=probs, name='DiscreteValuedDistribution')
    self._values = values

  @property
  def values(self) -> tf.Tensor:
    return self._values

  def _sample_n(self, n, seed=None) -> tf.Tensor:
    indices = super()._sample_n(n, seed=seed)
    return tf.gather(self.values, indices, axis=-1)

  def _mean(self) -> tf.Tensor:
    """Overrides the Categorical mean by incorporating category values."""
    return tf.reduce_sum(self.probs_parameter() * self.values, axis=-1)

  def _variance(self) -> tf.Tensor:
    """Overrides the Categorical variance by incorporating category values."""
    dist_squared = tf.square(tf.expand_dims(self.mean(), -1) - self.values)
    return tf.reduce_sum(self.probs_parameter() * dist_squared, axis=-1)
