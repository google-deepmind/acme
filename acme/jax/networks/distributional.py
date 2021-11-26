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

from typing import Any, List, Optional, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability
hk_init = hk.initializers
tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

_MIN_SCALE = 1e-4
Initializer = hk.initializers.Initializer


class CategoricalHead(hk.Module):
  """Module that produces a categorical distribution with the given number of values."""

  def __init__(
      self,
      num_values: Union[int, List[int]],
      dtype: Optional[Any] = jnp.int32,
      w_init: Optional[Initializer] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self._dtype = dtype
    self._logit_shape = num_values
    self._linear = hk.Linear(np.prod(num_values), w_init=w_init)

  def __call__(self, inputs: jnp.ndarray) -> tfd.Distribution:
    logits = self._linear(inputs)
    if not isinstance(self._logit_shape, int):
      logits = hk.Reshape(self._logit_shape)(logits)
    return tfd.Categorical(logits=logits, dtype=self._dtype)


class GaussianMixture(hk.Module):
  """Module that outputs a Gaussian Mixture Distribution."""

  def __init__(self,
               num_dimensions: int,
               num_components: int,
               multivariate: bool,
               init_scale: Optional[float] = None,
               append_singleton_event_dim: bool = False,
               reinterpreted_batch_ndims: Optional[int] = None,
               name: str = 'GaussianMixture'):
    """Initialization.

    Args:
      num_dimensions: dimensionality of the output distribution
      num_components: number of mixture components.
      multivariate: whether the resulting distribution is multivariate or not.
      init_scale: the initial scale for the Gaussian mixture components.
      append_singleton_event_dim: (univariate only) Whether to add an extra
        singleton dimension to the event shape.
      reinterpreted_batch_ndims: (univariate only) Number of batch dimensions to
        reinterpret as event dimensions.
      name: name of the module passed to snt.Module parent class.
    """
    super().__init__(name=name)

    self._num_dimensions = num_dimensions
    self._num_components = num_components
    self._multivariate = multivariate
    self._append_singleton_event_dim = append_singleton_event_dim
    self._reinterpreted_batch_ndims = reinterpreted_batch_ndims

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
      shape = [-1, self._num_components, self._num_dimensions]  # [B, C, D]
      # In this case, no need to reshape logits as they are in the correct shape
      # already, namely [batch_size, num_components].
    else:
      components_class = tfd.Normal
      shape = [-1, self._num_dimensions, self._num_components]  # [B, D, C]
      if self._append_singleton_event_dim:
        shape.insert(2, 1)  # [B, D, 1, C]
      logits = logits.reshape(shape)

    # Reshape the mixture's location and scale parameters appropriately.
    locs = locs.reshape(shape)
    scales = scales.reshape(shape)

    # Create the mixture distribution.
    distribution = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=components_class(loc=locs, scale=scales))

    if not self._multivariate:
      distribution = tfd.Independent(
          distribution,
          reinterpreted_batch_ndims=self._reinterpreted_batch_ndims)

    return distribution


class TanhTransformedDistribution(tfd.TransformedDistribution):
  """Distribution followed by tanh."""

  def __init__(self, distribution, threshold=.999, validate_args=False):
    """Initialize the distribution.

    Args:
      distribution: The distribution to transform.
      threshold: Clipping value of the action when computing the logprob.
      validate_args: Passed to super class.
    """
    super().__init__(
        distribution=distribution,
        bijector=tfp.bijectors.Tanh(),
        validate_args=validate_args)
    # Computes the log of the average probability distribution outside the
    # clipping range, i.e. on the interval [-inf, -atanh(threshold)] for
    # log_prob_left and [atanh(threshold), inf] for log_prob_right.
    self._threshold = threshold
    inverse_threshold = self.bijector.inverse(threshold)
    # average(pdf) = p/epsilon
    # So log(average(pdf)) = log(p) - log(epsilon)
    log_epsilon = jnp.log(1. - threshold)
    # Those 2 values are differentiable w.r.t. model parameters, such that the
    # gradient is defined everywhere.
    self._log_prob_left = self.distribution.log_cdf(
        -inverse_threshold) - log_epsilon
    self._log_prob_right = self.distribution.log_survival_function(
        inverse_threshold) - log_epsilon

  def log_prob(self, event):
    # Without this clip there would be NaNs in the inner tf.where and that
    # causes issues for some reasons.
    event = jnp.clip(event, -self._threshold, self._threshold)
    # The inverse image of {threshold} is the interval [atanh(threshold), inf]
    # which has a probability of "log_prob_right" under the given distribution.
    return jnp.where(
        event <= -self._threshold, self._log_prob_left,
        jnp.where(event >= self._threshold, self._log_prob_right,
                  super().log_prob(event)))

  def mode(self):
    return self.bijector.forward(self.distribution.mode())

  @classmethod
  def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
    td_properties = super()._parameter_properties(dtype,
                                                  num_classes=num_classes)
    del td_properties['bijector']
    return td_properties


class NormalTanhDistribution(hk.Module):
  """Module that produces a TanhTransformedDistribution distribution."""

  def __init__(self,
               num_dimensions: int,
               min_scale: float = 1e-3,
               w_init: hk_init.Initializer = hk_init.VarianceScaling(
                   1.0, 'fan_in', 'uniform'),
               b_init: hk_init.Initializer = hk_init.Constant(0.)):
    """Initialization.

    Args:
      num_dimensions: Number of dimensions of a distribution.
      min_scale: Minimum standard deviation.
      w_init: Initialization for linear layer weights.
      b_init: Initialization for linear layer biases.
    """
    super().__init__(name='Normal')
    self._min_scale = min_scale
    self._loc_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)
    self._scale_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)

  def __call__(self, inputs: jnp.ndarray) -> tfd.Distribution:
    loc = self._loc_layer(inputs)
    scale = self._scale_layer(inputs)
    scale = jax.nn.softplus(scale) + self._min_scale
    distribution = tfd.Normal(loc=loc, scale=scale)
    return tfd.Independent(
        TanhTransformedDistribution(distribution), reinterpreted_batch_ndims=1)


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


class DiscreteValued(hk.Module):
  """C51-style head.

  For each action, it produces the logits for a discrete distribution over
  atoms. Therefore, the returned logits represents several distributions, one
  for each action.
  """

  def __init__(
      self,
      num_actions: int,
      head_units: int = 512,
      num_atoms: int = 51,
      v_min: float = -1.0,
      v_max: float = 1.0,
  ):
    super().__init__('DiscreteValued')
    self._num_actions = num_actions
    self._num_atoms = num_atoms
    self._atoms = jnp.linspace(v_min, v_max, self._num_atoms)
    self._network = hk.nets.MLP([head_units, num_actions * num_atoms])

  def __call__(self, inputs: jnp.ndarray):
    q_logits = self._network(inputs)
    q_logits = jnp.reshape(q_logits, (-1, self._num_actions, self._num_atoms))
    q_dist = jax.nn.softmax(q_logits)
    q_values = jnp.sum(q_dist * self._atoms, axis=2)
    q_values = jax.lax.stop_gradient(q_values)
    return q_values, q_logits, self._atoms
