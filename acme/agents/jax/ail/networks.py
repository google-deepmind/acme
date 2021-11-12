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

"""Networks definitions for the BC agent.

AIRL network architecture follows https://arxiv.org/pdf/1710.11248.pdf.
"""
import dataclasses
import functools
from typing import Any, Callable, Generic, Iterable, Optional

from acme import specs
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax.imitation_learning_types import DirectRLNetworks
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np

# Function from discriminator logit to imitation reward.
ImitationRewardFn = Callable[[networks_lib.Logits], jnp.ndarray]
State = networks_lib.Params


@dataclasses.dataclass
class AILNetworks(Generic[DirectRLNetworks]):
  """AIL networks data class.

  Attributes:
    discriminator_network: Networks which takes as input:
      (observations, actions, next_observations, direct_rl_params)
      to return the logit of the discriminator.
      If the discriminator does not need direct_rl_params you can pass ().
    imitation_reward_fn: Function from logit of the discriminator to imitation
      reward.
    direct_rl_networks: Networks of the direct RL algorithm.
  """
  discriminator_network: networks_lib.FeedForwardNetwork
  imitation_reward_fn: ImitationRewardFn
  direct_rl_networks: DirectRLNetworks


def compute_ail_reward(discriminator_params: networks_lib.Params,
                       discriminator_state: State,
                       policy_params: Optional[networks_lib.Params],
                       transitions: types.Transition,
                       networks: AILNetworks) -> jnp.ndarray:
  """Computes the AIL reward for a given transition.

  Args:
    discriminator_params: Parameters of the discriminator network.
    discriminator_state: State of the discriminator network.
    policy_params: Parameters of the direct RL policy.
    transitions: Transitions to compute the reward for.
    networks: AIL networks.

  Returns:
    The rewards as an ndarray.
  """
  logits, _ = networks.discriminator_network.apply(
      discriminator_params,
      policy_params,
      discriminator_state,
      transitions,
      is_training=False,
      rng=None)
  return networks.imitation_reward_fn(logits)


class SpectralNormalizedLinear(hk.Module):
  """SpectralNormalizedLinear module.

  This is a Linear layer with a upper-bounded Lipschitz. It is used in iResNet.

  Reference:
    Behrmann et al. Invertible Residual Networks. ICML 2019.
    https://arxiv.org/pdf/1811.00995.pdf
  """

  def __init__(
      self,
      output_size: int,
      lipschitz_coeff: float,
      with_bias: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      name: Optional[str] = None,
  ):
    """Constructs the SpectralNormalizedLinear module.

    Args:
      output_size: Output dimensionality.
      lipschitz_coeff: Spectral normalization coefficient.
      with_bias: Whether to add a bias to the output.
      w_init: Optional initializer for weights. By default, uses random values
        from truncated normal, with stddev ``1 / sqrt(fan_in)``. See
        https://arxiv.org/abs/1502.03167v3.
      b_init: Optional initializer for bias. By default, zero.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.input_size = None
    self.output_size = output_size
    self.with_bias = with_bias
    self.w_init = w_init
    self.b_init = b_init or jnp.zeros
    self.lipschitz_coeff = lipschitz_coeff
    self.num_iterations = 100
    self.eps = 1e-6

  def get_normalized_weights(self,
                             weights: jnp.ndarray,
                             renormalize: bool = False) -> jnp.ndarray:

    def _l2_normalize(x, axis=None, eps=1e-12):
      return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)

    output_size = self.output_size
    dtype = weights.dtype
    assert output_size == weights.shape[-1]
    sigma = hk.get_state('sigma', (), init=jnp.ones)
    if renormalize:
      # Power iterations to compute spectral norm V*W*U^T.
      u = hk.get_state(
          'u', (1, output_size), dtype, init=hk.initializers.RandomNormal())
      for _ in range(self.num_iterations):
        v = _l2_normalize(jnp.matmul(u, weights.transpose()), eps=self.eps)
        u = _l2_normalize(jnp.matmul(v, weights), eps=self.eps)
      u = jax.lax.stop_gradient(u)
      v = jax.lax.stop_gradient(v)
      sigma = jnp.matmul(jnp.matmul(v, weights), jnp.transpose(u))[0, 0]
      hk.set_state('u', u)
      hk.set_state('v', v)
      hk.set_state('sigma', sigma)
    factor = jnp.maximum(1, sigma / self.lipschitz_coeff)
    return weights / factor

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Computes a linear transform of the input."""
    if not inputs.shape:
      raise ValueError('Input must not be scalar.')

    input_size = self.input_size = inputs.shape[-1]
    output_size = self.output_size
    dtype = inputs.dtype

    w_init = self.w_init
    if w_init is None:
      stddev = 1. / np.sqrt(self.input_size)
      w_init = hk.initializers.TruncatedNormal(stddev=stddev)
    w = hk.get_parameter('w', [input_size, output_size], dtype, init=w_init)
    w = self.get_normalized_weights(w, renormalize=True)

    out = jnp.dot(inputs, w)

    if self.with_bias:
      b = hk.get_parameter('b', [self.output_size], dtype, init=self.b_init)
      b = jnp.broadcast_to(b, out.shape)
      out = out + b

    return out


class DiscriminatorMLP(hk.Module):
  """A multi-layer perceptron module."""

  def __init__(
      self,
      hidden_layer_sizes: Iterable[int],
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      with_bias: bool = True,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
      input_dropout_rate: float = 0.,
      hidden_dropout_rate: float = 0.,
      spectral_normalization_lipschitz_coeff: Optional[float] = None,
      name: Optional[str] = None
  ):
    """Constructs an MLP.

    Args:
      hidden_layer_sizes: Hiddent layer sizes.
      w_init: Initializer for :class:`~haiku.Linear` weights.
      b_init: Initializer for :class:`~haiku.Linear` bias. Must be ``None`` if
        ``with_bias=False``.
      with_bias: Whether or not to apply a bias in each layer.
      activation: Activation function to apply between :class:`~haiku.Linear`
        layers. Defaults to ReLU.
      input_dropout_rate: Dropout on the input.
      hidden_dropout_rate: Dropout on the hidden layer outputs.
      spectral_normalization_lipschitz_coeff: If not None, the network will have
        spectral normalization with the given constant.
      name: Optional name for this module.

    Raises:
      ValueError: If ``with_bias`` is ``False`` and ``b_init`` is not ``None``.
    """
    if not with_bias and b_init is not None:
      raise ValueError('When with_bias=False b_init must not be set.')

    super().__init__(name=name)
    self._activation = activation
    self._input_dropout_rate = input_dropout_rate
    self._hidden_dropout_rate = hidden_dropout_rate
    layer_sizes = list(hidden_layer_sizes) + [1]

    if spectral_normalization_lipschitz_coeff is not None:
      layer_lipschitz_coeff = np.power(spectral_normalization_lipschitz_coeff,
                                       1. / len(layer_sizes))
      layer_module = functools.partial(
          SpectralNormalizedLinear,
          lipschitz_coeff=layer_lipschitz_coeff,
          w_init=w_init,
          b_init=b_init,
          with_bias=with_bias)
    else:
      layer_module = functools.partial(
          hk.Linear,
          w_init=w_init,
          b_init=b_init,
          with_bias=with_bias)

    layers = []
    for index, output_size in enumerate(layer_sizes):
      layers.append(
          layer_module(output_size=output_size, name=f'linear_{index}'))
    self._layers = tuple(layers)

  def __call__(
      self,
      inputs: jnp.ndarray,
      is_training: bool,
      rng: Optional[networks_lib.PRNGKey],
  ) -> networks_lib.Logits:
    rng = hk.PRNGSequence(rng) if rng is not None else None

    out = inputs
    for i, layer in enumerate(self._layers):
      if is_training:
        dropout_rate = (
            self._input_dropout_rate if i == 0 else self._hidden_dropout_rate)
        out = hk.dropout(next(rng), dropout_rate, out)
      out = layer(out)
      if i < len(self._layers) - 1:
        out = self._activation(out)

    return out


class DiscriminatorModule(hk.Module):
  """Discriminator module that concatenates its inputs."""

  def __init__(self,
               environment_spec: specs.EnvironmentSpec,
               use_action: bool,
               use_next_obs: bool,
               network_core: Callable[..., Any],
               observation_embedding: Callable[[networks_lib.Observation],
                                               jnp.ndarray] = lambda x: x,
               name='discriminator'):
    super().__init__(name=name)
    self._use_action = use_action
    self._environment_spec = environment_spec
    self._use_next_obs = use_next_obs
    self._network_core = network_core
    self._observation_embedding = observation_embedding

  def __call__(self, observations: networks_lib.Observation,
               actions: networks_lib.Action,
               next_observations: networks_lib.Observation, is_training: bool,
               rng: networks_lib.PRNGKey) -> networks_lib.Logits:
    observations = self._observation_embedding(observations)
    if self._use_next_obs:
      next_observations = self._observation_embedding(next_observations)
      data = jnp.concatenate([observations, next_observations], axis=-1)
    else:
      data = observations
    if self._use_action:
      action_spec = self._environment_spec.actions
      if isinstance(action_spec, specs.DiscreteArray):
        actions = jax.nn.one_hot(actions,
                                 action_spec.num_values)
      data = jnp.concatenate([data, actions], axis=-1)
    output = self._network_core(data, is_training, rng)
    output = jnp.squeeze(output, axis=-1)
    return output


class AIRLModule(hk.Module):
  """AIRL Module."""

  def __init__(self,
               environment_spec: specs.EnvironmentSpec,
               use_action: bool,
               use_next_obs: bool,
               discount: float,
               g_core: Callable[..., Any],
               h_core: Callable[..., Any],
               observation_embedding: Callable[[networks_lib.Observation],
                                               jnp.ndarray] = lambda x: x,
               name='airl'):
    super().__init__(name=name)
    self._environment_spec = environment_spec
    self._use_action = use_action
    self._use_next_obs = use_next_obs
    self._discount = discount
    self._g_core = g_core
    self._h_core = h_core
    self._observation_embedding = observation_embedding

  def __call__(self, observations: networks_lib.Observation,
               actions: networks_lib.Action,
               next_observations: networks_lib.Observation,
               is_training: bool,
               rng: networks_lib.PRNGKey) -> networks_lib.Logits:
    g_output = DiscriminatorModule(
        environment_spec=self._environment_spec,
        use_action=self._use_action,
        use_next_obs=self._use_next_obs,
        network_core=self._g_core,
        observation_embedding=self._observation_embedding,
        name='airl_g')(observations, actions, next_observations, is_training,
                       rng)
    h_module = DiscriminatorModule(
        environment_spec=self._environment_spec,
        use_action=False,
        use_next_obs=False,
        network_core=self._h_core,
        observation_embedding=self._observation_embedding,
        name='airl_h')
    return (g_output + self._discount * h_module(next_observations, (),
                                                 (), is_training, rng) -
            h_module(observations, (), (), is_training, rng))


# TODO(eorsini): Manipulate FeedForwardNetworks instead of transforms to
# increase compatibility with Flax.
def make_discriminator(
    environment_spec: specs.EnvironmentSpec,
    discriminator_transformed: hk.TransformedWithState,
    logpi_fn: Optional[Callable[
        [networks_lib.Params, networks_lib.Observation, networks_lib.Action],
        jnp.ndarray]] = None
) -> networks_lib.FeedForwardNetwork:
  """Creates the discriminator network.

  Args:
    environment_spec: Environment spec
    discriminator_transformed: Haiku transformed of the discriminator.
    logpi_fn: If the policy logpi function is provided, its output will be
      removed from the discriminator logit.

  Returns:
    The network.
  """

  def apply_fn(params: hk.Params,
               policy_params: networks_lib.Params,
               state: hk.State,
               transitions: types.Transition,
               is_training: bool,
               rng: networks_lib.PRNGKey) -> networks_lib.Logits:
    output, state = discriminator_transformed.apply(
        params, state, transitions.observation, transitions.action,
        transitions.next_observation, is_training, rng)
    if logpi_fn is not None:
      logpi = logpi_fn(policy_params, transitions.observation,
                       transitions.action)

      # Quick Maths:
      # D = exp(output)/(exp(output) + pi(a|s))
      # logit(D) = log(D/(1-D)) = log(exp(output)/pi(a|s))
      # logit(D) = output - logpi
      return output - logpi, state
    return output, state

  dummy_obs = utils.zeros_like(environment_spec.observations)
  dummy_obs = utils.add_batch_dim(dummy_obs)
  dummy_actions = utils.zeros_like(environment_spec.actions)
  dummy_actions = utils.add_batch_dim(dummy_actions)

  return networks_lib.FeedForwardNetwork(
      # pylint: disable=g-long-lambda
      init=lambda rng: discriminator_transformed.init(
          rng, dummy_obs, dummy_actions, dummy_obs, False, rng),
      apply=apply_fn)
