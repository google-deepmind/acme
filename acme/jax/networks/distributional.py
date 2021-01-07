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
hk_init = hk.initializers
tfd = tensorflow_probability.experimental.substrates.jax.distributions

_MIN_SCALE = 1e-4


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


class GaussianMixture(hk.Module):
  """Module that outputs a Gaussian Mixture Distribution."""

  def __init__(self,
               num_dimensions: int,
               num_components: int,
               multivariate: bool,
               init_scale: Optional[float] = None,
               name: str = 'GaussianMixture'):
    """Initialization.

    Args:
      num_dimensions: dimensionality of the output distribution
      num_components: number of mixture components.
      multivariate: whether the resulting distribution is multivariate or not.
      init_scale: the initial scale for the Gaussian mixture components.
      name: name of the module passed to snt.Module parent class.
    """
    super().__init__(name=name)

    self._num_dimensions = num_dimensions
    self._num_components = num_components
    self._multivariate = multivariate

    if init_scale is not None:
      self._scale_factor = init_scale / jax.nn.softplus(0.)
    else:
      self._scale_factor = 1.0  # Corresponds to init_scale = softplus(0).

  def __call__(self,
               inputs: jnp.ndarray,
               low_noise_policy: bool = False) -> tfd.Distribution:
    """Run the networks through inputs.

    Args:
      inputs: hidden activations of the policy network body.
      low_noise_policy: whether to set vanishingly small scales for each
        component. If this flag is set to True, the policy is effectively run
        without Gaussian noise.

    Returns:
      Mixture Gaussian distribution.
    """

    # Define the weight initializer.
    w_init = hk.initializers.VarianceScaling(scale=1e-5)

    # Create a layer that outputs the unnormalized log-weights.
    if self._multivariate:
      logits_size = self._num_components
    else:
      logits_size = self._num_dimensions * self._num_components
    logit_layer = hk.Linear(logits_size, w_init=w_init)

    # Create two layers that outputs a location and a scale, respectively, for
    # each dimension and each component.
    loc_layer = hk.Linear(
        self._num_dimensions * self._num_components, w_init=w_init)
    scale_layer = hk.Linear(
        self._num_dimensions * self._num_components, w_init=w_init)

    # Compute logits, locs, and scales if necessary.
    logits = logit_layer(inputs)
    locs = loc_layer(inputs)

    # When a low_noise_policy is requested, set the scales to its minimum value.
    if low_noise_policy:
      scales = jnp.full(locs.shape, _MIN_SCALE)
    else:
      scales = scale_layer(inputs)
      scales = self._scale_factor * jax.nn.softplus(scales) + _MIN_SCALE

    if self._multivariate:
      components_class = tfd.MultivariateNormalDiag
      shape = [-1, self._num_components, self._num_dimensions]
      # In this case, no need to reshape logits as they are in the correct shape
      # already, namely [batch_size, num_components].
    else:
      components_class = tfd.Normal
      shape = [-1, self._num_dimensions, self._num_components]
      logits = logits.reshape(shape)

    # Reshape the mixture's location and scale parameters appropriately.
    locs = locs.reshape(shape)
    scales = scales.reshape(shape)

    # Create the mixture distribution.
    distribution = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=components_class(loc=locs, scale=scales))

    if not self._multivariate:
      distribution = tfd.Independent(distribution)

    return distribution


class MultivariateNormalDiagHead(hk.Module):
  """Module that produces a tfd.MultivariateNormalDiag distribution."""

  def __init__(self,
               num_dimensions: int,
               min_scale: float = 1e-6,
               w_init: hk_init.Initializer = hk_init.VarianceScaling(1e-4),
               b_init: hk_init.Initializer = hk_init.Constant(0.)):
    """Initialization.

    Args:
      num_dimensions: Number of dimensions of MVN distribution.
      min_scale: Minimum standard deviation.
      w_init: Initialization for linear layer weights.
      b_init: Initialization for linear layer biases.
    """
    super().__init__(name='MultivariateNormalDiagHead')
    self._min_scale = min_scale
    self._loc_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)
    self._scale_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)

  def __call__(self, inputs: jnp.ndarray) -> tfd.Distribution:
    loc = self._loc_layer(inputs)
    scale = self._scale_layer(inputs)
    scale = jax.nn.softplus(scale) + self._min_scale
    return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)


class CategoricalValueHead(hk.Module):
  """Network head that produces a categorical distribution and value."""

  def __init__(
      self,
      num_values: int,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._logit_layer = hk.Linear(num_values)
    self._value_layer = hk.Linear(1)

  def __call__(self, inputs: jnp.ndarray):
    logits = self._logit_layer(inputs)
    value = jnp.squeeze(self._value_layer(inputs), axis=-1)
    return (tfd.Categorical(logits=logits), value)
