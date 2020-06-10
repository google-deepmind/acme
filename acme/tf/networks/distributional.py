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

"""Distributional modules: these are modules that return a tfd.Distribution.

There are useful modules in `acme.networks.stochastic` to either sample or
take the mean of these distributions.
"""

import types
from acme.tf.networks import distributions as ad
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
snt_init = snt.initializers


class DiscreteValuedHead(snt.Module):
  """Represents a parameterized discrete valued distribution.

  The returned distribution is essentially a `tfd.Categorical`, but one which
  knows its support and so can compute the mean value.
  """

  def __init__(self,
               vmin: float,
               vmax: float,
               num_atoms: int,
               w_init: snt.initializers.Initializer = None,
               b_init: snt.initializers.Initializer = None):
    """Initialization.

    Args:
      vmin: Minimum of the value range
      vmax: Maximum of the value range
      num_atoms: The atom values associated with each bin.
      w_init: Initialization for linear layer weights.
      b_init: Initialization for linear layer biases.
    """
    super().__init__(name='DiscreteValuedHead')
    self._distributional_layer = snt.Linear(
        num_atoms, w_init=w_init, b_init=b_init)
    self._values = tf.linspace(vmin, vmax, num_atoms)

  def __call__(self, inputs: tf.Tensor) -> tfd.Distribution:
    logits = self._distributional_layer(inputs)
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


class GaussianMixtureHead(snt.Module):
  """Head which outputs a Mixture of Gaussians Distribution."""

  def __init__(self,
               num_dimensions: int,
               num_mixtures: int = 5):
    """Create an mixture of Gaussian actor head.

    Args:
      num_dimensions: dimensionality of the output distribution. Each dimension
        is going to be an independent 1d GMM model.
      num_mixtures: number of mixture components.
    """
    super().__init__(name='GaussianMixtureHead')

    self._num_dimensions = num_dimensions
    self._num_mixtures = num_mixtures

    initializer = tf.initializers.VarianceScaling(
        distribution='uniform', mode='fan_out', scale=0.333)
    # For every dimension (self._num_dimensions) and every component
    # (self._num_mixtures), the network should output a location, a scale, and
    # a log mixture probability (3 scalars).
    output_sizes = [self._num_dimensions * self._num_mixtures * 3]
    self._mlp = snt.nets.MLP(
        output_sizes=output_sizes,
        w_init=initializer,
        activation=tf.nn.relu,
        activate_final=False)

  def __call__(self, inputs: tf.Tensor) -> tfd.Distribution:
    """Run the networks through inputs.

    Args:
      inputs: hidden activations of the policy network body.

    Returns:
      Gaussian Mixture Model distribution.
    """

    # Multiplied by 3 because we need location, scale, and mixture logits.
    action_output = tf.reshape(
        self._mlp(inputs), [-1, self._num_dimensions, self._num_mixtures * 3])

    locs, scales, logits = tf.split(action_output, 3, -1)

    scales = tf.nn.softplus(scales) + 0.01

    components_distribution = tfd.Normal(loc=locs, scale=scales)
    distribution = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=logits),
        components_distribution=components_distribution)

    distribution = tfd.Independent(distribution)

    return distribution


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
