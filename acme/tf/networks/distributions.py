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

from typing import Optional
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


@tfp.experimental.register_composite
class DiscreteValuedDistribution(tfd.Categorical):
  """This is a generalization of a categorical distribution.

  The support for the DiscreteValued distribution can be any real valued range,
  whereas the categorical distribution has support [0, n_categories - 1] or
  [1, n_categories]. This generalization allows us to take the mean of the
  distribution over its support.
  """

  def __init__(self,
               values: tf.Tensor,
               logits: Optional[tf.Tensor] = None,
               probs: Optional[tf.Tensor] = None,
               name: str = 'DiscreteValuedDistribution'):
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
      name: Name of the distribution object.
    """
    self._values = tf.convert_to_tensor(values)
    shape_strings = [f'D{i}' for i, _ in enumerate(values.shape)]

    if logits is not None:
      logits = tf.convert_to_tensor(logits)
      tf.debugging.assert_shapes([(values, shape_strings),
                                  (logits, [..., *shape_strings])])
    if probs is not None:
      probs = tf.convert_to_tensor(probs)
      tf.debugging.assert_shapes([(values, shape_strings),
                                  (probs, [..., *shape_strings])])

    super().__init__(logits=logits, probs=probs, name=name)

    self._parameters = dict(values=values,
                            logits=logits,
                            probs=probs,
                            name=name)

  @property
  def values(self) -> tf.Tensor:
    return self._values

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        values=tfp.util.ParameterProperties(event_ndims=None),
        logits=tfp.util.ParameterProperties(
            event_ndims=lambda self: self.values.shape.rank),
        probs=tfp.util.ParameterProperties(
            event_ndims=lambda self: self.values.shape.rank,
            is_preferred=False))

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

  def _event_shape(self):
    # Omit the atoms axis, to return just the shape of a single (i.e. unbatched)
    # sample value.
    return self._values.shape[:-1]

  def _event_shape_tensor(self):
    return tf.shape(self._values)[:-1]
