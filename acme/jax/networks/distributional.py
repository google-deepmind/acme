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

"""Haiku modules that output tfd.Distributions."""

from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow_probability
tfd = tensorflow_probability.experimental.substrates.jax.distributions


class CategoricalHead(hk.Module):
  """Module that produces a categorical distribution with the given number of values."""

  def __init__(
      self,
      num_values: int,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._linear = hk.Linear(num_values)

  def __call__(self, inputs: jnp.ndarray) -> tfd.Distribution:
    logits = self._linear(inputs)
    return tfd.Categorical(logits=logits)


class MultivariateNormalDiagHead(hk.Module):
  """Module that produces a tfd.MultivariateNormalDiag distribution."""

  def __init__(self, num_dimensions: int, min_scale: float = 1e-6):
    """Initialization.

    Args:
      num_dimensions: Number of dimensions of MVN distribution.
      min_scale: Minimum standard deviation.
    """
    super().__init__(name='MultivariateNormalDiagHead')
    self._min_scale = min_scale
    self._loc_layer = hk.Linear(num_dimensions)
    self._scale_layer = hk.Linear(num_dimensions)

  def __call__(self, inputs: jnp.ndarray) -> tfd.Distribution:
    loc = self._loc_layer(inputs)
    scale = self._scale_layer(inputs)
    scale = jax.nn.softplus(scale) + self._min_scale
    return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
