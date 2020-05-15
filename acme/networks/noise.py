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

"""Noise layers (for exploration)."""

import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class ClippedGaussian(snt.Module):
  """Sonnet module for adding clipped Gaussian noise to each output."""

  def __init__(self, stddev: float, name: str = 'clipped_gaussian'):
    super().__init__(name=name)
    self._noise = tfd.Normal(loc=0., scale=stddev)

  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    output = inputs + self._noise.sample(inputs.shape)
    output = tf.clip_by_value(output, -1.0, 1.0)

    return output
