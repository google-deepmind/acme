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

"""Distributional modules: these are modules that return a tfd.Distribution.

There are useful modules in `acme.networks.stochastic` to either sample or
take the mean of these distributions.
"""

import types
from typing import Optional, Union
from absl import logging
from acme.tf.networks import distributions as ad
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
snt_init = snt.initializers

_MIN_SCALE = 1e-4


class DiscreteValuedHead(snt.Module):
  """Represents a parameterized discrete valued distribution.

  The returned distribution is essentially a `tfd.Categorical`, but one which
  knows its support and so can compute the mean value.
  """

  def __init__(self,
               vmin: Union[float, np.ndarray, tf.Tensor],
               vmax: Union[float, np.ndarray, tf.Tensor],
               num_atoms: int,
               w_init: Optional[snt.initializers.Initializer] = None,
               b_init: Optional[snt.initializers.Initializer] = None):
    """Initialization.

    If vmin and vmax have shape S, this will store the category values as a
    Tensor of shape (S*, num_atoms).

    Args:
      vmin: Minimum of the value range
      vmax: Maximum of the value range
      num_atoms: The atom values associated with each bin.
      w_init: Initialization for linear layer weights.
      b_init: Initialization for linear layer biases.
    """
    super().__init__(name='DiscreteValuedHead')
    vmin = tf.convert_to_tensor(vmin)
    vmax = tf.convert_to_tensor(vmax)
    self._values = tf.linspace(vmin, vmax, num_atoms, axis=-1)
    self._distributional_layer = snt.Linear(tf.size(self._values),
                                            w_init=w_init,
                                            b_init=b_init)

  def __call__(self, inputs: tf.Tensor) -> tfd.Distribution:
    logits = self._distributional_layer(inputs)
    logits = tf.reshape(logits,
                        tf.concat([tf.shape(logits)[:1],  # batch size
                                   tf.shape(self._values)],
                                  axis=0))
    values = tf.cast(self._values, logits.dtype)

    return ad.DiscreteValuedDistribution(values=values, logits=logits)


class MultivariateNormalDiagHead(snt.Module):
  """Module that produces a multivariate normal distribution using tfd.Independent or tfd.MultivariateNormalDiag."""

  def __init__(
      self,
      num_dimensions: int,
      init_scale: float = 0.3,
      min_scale: float = 1e-6,
      tanh_mean: bool = False,
      fixed_scale: bool = False,
      use_tfd_independent: bool = False,
      w_init: snt_init.Initializer = tf.initializers.VarianceScaling(1e-4),
      b_init: snt_init.Initializer = tf.initializers.Zeros()):
    """Initialization.

    Args:
      num_dimensions: Number of dimensions of MVN distribution.
      init_scale: Initial standard deviation.
      min_scale: Minimum standard deviation.
      tanh_mean: Whether to transform the mean (via tanh) before passing it to
        the distribution.
      fixed_scale: Whether to use a fixed variance.
      use_tfd_independent: Whether to use tfd.Independent or
        tfd.MultivariateNormalDiag class
      w_init: Initialization for linear layer weights.
      b_init: Initialization for linear layer biases.
    """
    super().__init__(name='MultivariateNormalDiagHead')
    self._init_scale = init_scale
    self._min_scale = min_scale
    self._tanh_mean = tanh_mean
    self._mean_layer = snt.Linear(num_dimensions, w_init=w_init, b_init=b_init)
    self._fixed_scale = fixed_scale

    if not fixed_scale:
      self._scale_layer = snt.Linear(
          num_dimensions, w_init=w_init, b_init=b_init)
    self._use_tfd_independent = use_tfd_independent

  def __call__(self, inputs: tf.Tensor) -> tfd.Distribution:
    zero = tf.constant(0, dtype=inputs.dtype)
    mean = self._mean_layer(inputs)

    if self._fixed_scale:
      scale = tf.ones_like(mean) * self._init_scale
    else:
      scale = tf.nn.softplus(self._scale_layer(inputs))
      scale *= self._init_scale / tf.nn.softplus(zero)
      scale += self._min_scale

    # Maybe transform the mean.
    if self._tanh_mean:
      mean = tf.tanh(mean)

    if self._use_tfd_independent:
      dist = tfd.Independent(tfd.Normal(loc=mean, scale=scale))
    else:
      dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=scale)

    return dist


class GaussianMixture(snt.Module):
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
      self._scale_factor = init_scale / tf.nn.softplus(0.)
    else:
      self._scale_factor = 1.0  # Corresponds to init_scale = softplus(0).

    # Define the weight initializer.
    w_init = tf.initializers.VarianceScaling(1e-5)

    # Create a layer that outputs the unnormalized log-weights.
    if self._multivariate:
      logits_size = self._num_components
    else:
      logits_size = self._num_dimensions * self._num_components
    self._logit_layer = snt.Linear(logits_size, w_init=w_init)

    # Create two layers that outputs a location and a scale, respectively, for
    # each dimension and each component.
    self._loc_layer = snt.Linear(
        self._num_dimensions * self._num_components, w_init=w_init)
    self._scale_layer = snt.Linear(
        self._num_dimensions * self._num_components, w_init=w_init)

  def __call__(self,
               inputs: tf.Tensor,
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

    # Compute logits, locs, and scales if necessary.
    logits = self._logit_layer(inputs)
    locs = self._loc_layer(inputs)

    # When a low_noise_policy is requested, set the scales to its minimum value.
    if low_noise_policy:
      scales = tf.fill(locs.shape, _MIN_SCALE)
    else:
      scales = self._scale_layer(inputs)
      scales = self._scale_factor * tf.nn.softplus(scales) + _MIN_SCALE

    if self._multivariate:
      shape = [-1, self._num_components, self._num_dimensions]
      # Reshape the mixture's location and scale parameters appropriately.
      locs = tf.reshape(locs, shape)
      scales = tf.reshape(scales, shape)
      # In this case, no need to reshape logits as they are in the correct shape
      # already, namely [batch_size, num_components].
      components_distribution = tfd.MultivariateNormalDiag(
          loc=locs, scale_diag=scales)
    else:
      shape = [-1, self._num_dimensions, self._num_components]
      # Reshape the mixture's location and scale parameters appropriately.
      locs = tf.reshape(locs, shape)
      scales = tf.reshape(scales, shape)
      components_distribution = tfd.Normal(loc=locs, scale=scales)
      logits = tf.reshape(logits, shape)

    # Create the mixture distribution.
    distribution = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=components_distribution)

    if not self._multivariate:
      distribution = tfd.Independent(distribution)

    return distribution


class UnivariateGaussianMixture(GaussianMixture):
  """Head which outputs a Mixture of Gaussians Distribution."""

  def __init__(self,
               num_dimensions: int,
               num_components: int = 5,
               init_scale: Optional[float] = None,
               num_mixtures: Optional[int] = None):
    """Create an mixture of Gaussian actor head.

    Args:
      num_dimensions: dimensionality of the output distribution. Each dimension
        is going to be an independent 1d GMM model.
      num_components: number of mixture components.
      init_scale: the initial scale for the Gaussian mixture components.
      num_mixtures: deprecated argument which overwrites num_components.
    """
    if num_mixtures is not None:
      logging.warning("""the num_mixtures parameter has been deprecated; use
                    num_components instead; the value of num_components is being
                    ignored""")
      num_components = num_mixtures
    super().__init__(num_dimensions=num_dimensions,
                     num_components=num_components,
                     multivariate=False,
                     init_scale=init_scale,
                     name='UnivariateGaussianMixture')


class MultivariateGaussianMixture(GaussianMixture):
  """Head which outputs a mixture of multivariate Gaussians distribution."""

  def __init__(self,
               num_dimensions: int,
               num_components: int = 5,
               init_scale: Optional[float] = None):
    """Initialization.

    Args:
      num_dimensions: dimensionality of the output distribution
        (also the dimensionality of the multivariate Gaussian model).
      num_components: number of mixture components.
      init_scale: the initial scale for the Gaussian mixture components.
    """
    super().__init__(num_dimensions=num_dimensions,
                     num_components=num_components,
                     multivariate=True,
                     init_scale=init_scale,
                     name='MultivariateGaussianMixture')


class ApproximateMode(snt.Module):
  """Override the mode function of the distribution.

  For non-constant Jacobian transformed distributions, the mode is non-trivial
  to compute, so for these distributions the mode function is not supported in
  TFP. A frequently used approximation is to forward transform the mode of the
  untransformed distribution.

  Otherwise (an untransformed distribution or a transformed distribution with a
  constant Jacobian), this is a no-op.
  """

  def __call__(self, inputs: tfd.Distribution) -> tfd.Distribution:
    if isinstance(inputs, tfd.TransformedDistribution):
      if not inputs.bijector.is_constant_jacobian:
        def _mode(self, **kwargs):
          distribution_kwargs, bijector_kwargs = self._kwargs_split_fn(kwargs)
          x = self.distribution.mode(**distribution_kwargs)
          y = self.bijector.forward(x, **bijector_kwargs)
          return y
        inputs._mode = types.MethodType(_mode, inputs)
    return inputs
