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

"""PPO network definitions."""

import dataclasses
from typing import Callable, NamedTuple, Optional, Sequence

from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

EntropyFn = Callable[
    [networks_lib.Params, networks_lib.PRNGKey], networks_lib.Entropy
]


class MVNDiagParams(NamedTuple):
  """Parameters for a diagonal multi-variate normal distribution."""
  loc: jnp.ndarray
  scale_diag: jnp.ndarray


class TanhNormalParams(NamedTuple):
  """Parameters for a tanh squashed diagonal MVN distribution."""
  loc: jnp.ndarray
  scale: jnp.ndarray


class CategoricalParams(NamedTuple):
  """Parameters for a categorical distribution."""
  logits: jnp.ndarray


class PPOParams(NamedTuple):
  model_params: networks_lib.Params
  # Using float32 as it covers a larger range than int32. If using int64 we
  # would need to do jax_enable_x64.
  num_sgd_steps: jnp.float32


@dataclasses.dataclass
class PPONetworks:
  """Network and pure functions for the PPO agent.

  If 'network' returns tfd.Distribution, you can use make_ppo_networks() to
  create this object properly.
  If one is building this object manually, one has a freedom to make 'network'
  object return anything that is later being passed as input to
  log_prob/entropy/sample functions to perform the corresponding computations.
  An example scenario where you would want to do this due to
  tfd.Distribution not playing nice with jax.vmap. Please refer to the
  make_continuous_networks() for an example where the network does not return a
  tfd.Distribution object.
  """
  network: networks_lib.FeedForwardNetwork
  log_prob: networks_lib.LogProbFn
  entropy: EntropyFn
  sample: networks_lib.SampleFn
  sample_eval: Optional[networks_lib.SampleFn] = None


def make_inference_fn(
    ppo_networks: PPONetworks,
    evaluation: bool = False) -> actor_core_lib.FeedForwardPolicyWithExtra:
  """Returns a function to be used for inference by a PPO actor."""

  def inference(
      params: networks_lib.Params,
      key: networks_lib.PRNGKey,
      observations: networks_lib.Observation,
  ):
    dist_params, _ = ppo_networks.network.apply(params.model_params,
                                                observations)
    if evaluation and ppo_networks.sample_eval:
      actions = ppo_networks.sample_eval(dist_params, key)
    else:
      actions = ppo_networks.sample(dist_params, key)
    if evaluation:
      return actions, {}
    log_prob = ppo_networks.log_prob(dist_params, actions)
    extras = {
        'log_prob': log_prob,
        # Add batch dimension.
        'params_num_sgd_steps': params.num_sgd_steps[None, ...]
    }
    return actions, extras

  return inference


def make_networks(
    spec: specs.EnvironmentSpec, hidden_layer_sizes: Sequence[int] = (256, 256)
) -> PPONetworks:
  if isinstance(spec.actions, specs.DiscreteArray):
    return make_discrete_networks(spec, hidden_layer_sizes)
  else:
    return make_continuous_networks(
        spec,
        policy_layer_sizes=hidden_layer_sizes,
        value_layer_sizes=hidden_layer_sizes)


def make_ppo_networks(network: networks_lib.FeedForwardNetwork) -> PPONetworks:
  """Constructs a PPONetworks instance from the given FeedForwardNetwork.

  This method assumes that the network returns a tfd.Distribution. Sometimes it
  may be preferable to have networks that do not return tfd.Distribution
  objects, for example, due to tfd.Distribution not playing nice with jax.vmap.
  Please refer to the make_continuous_networks() for an example where the
  network does not return a tfd.Distribution object.

  Args:
    network: a transformed Haiku network that takes in observations and returns
      the action distribution and value.

  Returns:
    A PPONetworks instance with pure functions wrapping the input network.
  """
  return PPONetworks(
      network=network,
      log_prob=lambda distribution, action: distribution.log_prob(action),
      entropy=lambda distribution, key=None: distribution.entropy(),
      sample=lambda distribution, key: distribution.sample(seed=key),
      sample_eval=lambda distribution, key: distribution.mode())


def make_mvn_diag_ppo_networks(
    network: networks_lib.FeedForwardNetwork) -> PPONetworks:
  """Constructs a PPONetworks for MVN Diag policy from the FeedForwardNetwork.

  Args:
    network: a transformed Haiku network (or equivalent in other libraries) that
      takes in observations and returns the action distribution and value.

  Returns:
    A PPONetworks instance with pure functions wrapping the input network.
  """

  def log_prob(params: MVNDiagParams, action):
    return tfd.MultivariateNormalDiag(
        loc=params.loc, scale_diag=params.scale_diag).log_prob(action)

  def entropy(
      params: MVNDiagParams, key: networks_lib.PRNGKey
  ) -> networks_lib.Entropy:
    del key
    return tfd.MultivariateNormalDiag(
        loc=params.loc, scale_diag=params.scale_diag).entropy()

  def sample(params: MVNDiagParams, key: networks_lib.PRNGKey):
    return tfd.MultivariateNormalDiag(
        loc=params.loc, scale_diag=params.scale_diag).sample(seed=key)

  def sample_eval(params: MVNDiagParams, key: networks_lib.PRNGKey):
    del key
    return tfd.MultivariateNormalDiag(
        loc=params.loc, scale_diag=params.scale_diag).mode()

  return PPONetworks(
      network=network,
      log_prob=log_prob,
      entropy=entropy,
      sample=sample,
      sample_eval=sample_eval)


def make_tanh_normal_ppo_networks(
    network: networks_lib.FeedForwardNetwork) -> PPONetworks:
  """Constructs a PPONetworks for Tanh MVN Diag policy from the FeedForwardNetwork.

  Args:
    network: a transformed Haiku network (or equivalent in other libraries) that
      takes in observations and returns the action distribution and value.

  Returns:
    A PPONetworks instance with pure functions wrapping the input network.
  """

  def build_distribution(params: TanhNormalParams):
    distribution = tfd.Normal(loc=params.loc, scale=params.scale)
    distribution = tfd.Independent(
        networks_lib.TanhTransformedDistribution(distribution),
        reinterpreted_batch_ndims=1)
    return distribution

  def log_prob(params: TanhNormalParams, action):
    distribution = build_distribution(params)
    return distribution.log_prob(action)

  def entropy(
      params: TanhNormalParams, key: networks_lib.PRNGKey
  ) -> networks_lib.Entropy:
    distribution = build_distribution(params)
    return distribution.entropy(seed=key)

  def sample(params: TanhNormalParams, key: networks_lib.PRNGKey):
    distribution = build_distribution(params)
    return distribution.sample(seed=key)

  def sample_eval(params: TanhNormalParams, key: networks_lib.PRNGKey):
    del key
    distribution = build_distribution(params)
    return distribution.mode()

  return PPONetworks(
      network=network,
      log_prob=log_prob,
      entropy=entropy,
      sample=sample,
      sample_eval=sample_eval)


def make_discrete_networks(
    environment_spec: specs.EnvironmentSpec,
    hidden_layer_sizes: Sequence[int] = (512,),
    use_conv: bool = True,
) -> PPONetworks:
  """Creates networks used by the agent for discrete action environments.

  Args:
    environment_spec: Environment spec used to define number of actions.
    hidden_layer_sizes: Network definition.
    use_conv: Whether to use a conv or MLP feature extractor.
  Returns:
    PPONetworks
  """

  num_actions = environment_spec.actions.num_values

  def forward_fn(inputs):
    layers = []
    if use_conv:
      layers.extend([networks_lib.AtariTorso()])
    layers.extend([hk.nets.MLP(hidden_layer_sizes, activate_final=True)])
    trunk = hk.Sequential(layers)
    h = utils.batch_concat(inputs)
    h = trunk(h)
    logits = hk.Linear(num_actions)(h)
    values = hk.Linear(1)(h)
    values = jnp.squeeze(values, axis=-1)
    return (CategoricalParams(logits=logits), values)

  forward_fn = hk.without_apply_rng(hk.transform(forward_fn))
  dummy_obs = utils.zeros_like(environment_spec.observations)
  dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.
  network = networks_lib.FeedForwardNetwork(
      lambda rng: forward_fn.init(rng, dummy_obs), forward_fn.apply)
  # Create PPONetworks to add functionality required by the agent.
  return make_categorical_ppo_networks(network)  # pylint:disable=undefined-variable


def make_categorical_ppo_networks(
    network: networks_lib.FeedForwardNetwork) -> PPONetworks:
  """Constructs a PPONetworks for Categorical Policy from FeedForwardNetwork.

  Args:
    network: a transformed Haiku network (or equivalent in other libraries) that
      takes in observations and returns the action distribution and value.

  Returns:
    A PPONetworks instance with pure functions wrapping the input network.
  """

  def log_prob(params: CategoricalParams, action):
    return tfd.Categorical(logits=params.logits).log_prob(action)

  def entropy(
      params: CategoricalParams, key: networks_lib.PRNGKey
  ) -> networks_lib.Entropy:
    del key
    return tfd.Categorical(logits=params.logits).entropy()

  def sample(params: CategoricalParams, key: networks_lib.PRNGKey):
    return tfd.Categorical(logits=params.logits).sample(seed=key)

  def sample_eval(params: CategoricalParams, key: networks_lib.PRNGKey):
    del key
    return tfd.Categorical(logits=params.logits).mode()

  return PPONetworks(
      network=network,
      log_prob=log_prob,
      entropy=entropy,
      sample=sample,
      sample_eval=sample_eval)


def make_continuous_networks(
    environment_spec: specs.EnvironmentSpec,
    policy_layer_sizes: Sequence[int] = (64, 64),
    value_layer_sizes: Sequence[int] = (64, 64),
    use_tanh_gaussian_policy: bool = True,
) -> PPONetworks:
  """Creates PPONetworks to be used for continuous action environments."""

  # Get total number of action dimensions from action spec.
  num_dimensions = np.prod(environment_spec.actions.shape, dtype=int)

  def forward_fn(inputs: networks_lib.Observation):

    def _policy_network(obs: networks_lib.Observation):
      h = utils.batch_concat(obs)
      h = hk.nets.MLP(policy_layer_sizes, activate_final=True)(h)

      # tfd distributions have a weird bug in jax when vmapping is used, so the
      # safer implementation in general is for the policy network to output the
      # distribution parameters, and for the distribution to be constructed
      # in a method such as make_ppo_networks above
      if not use_tanh_gaussian_policy:
        # Following networks_lib.MultivariateNormalDiagHead
        init_scale = 0.3
        min_scale = 1e-6
        w_init = hk.initializers.VarianceScaling(1e-4)
        b_init = hk.initializers.Constant(0.)
        loc_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)
        scale_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)

        loc = loc_layer(h)
        scale = jax.nn.softplus(scale_layer(h))
        scale *= init_scale / jax.nn.softplus(0.)
        scale += min_scale

        return MVNDiagParams(loc=loc, scale_diag=scale)

      # Following networks_lib.NormalTanhDistribution
      min_scale = 1e-3
      w_init = hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform')
      b_init = hk.initializers.Constant(0.)
      loc_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)
      scale_layer = hk.Linear(num_dimensions, w_init=w_init, b_init=b_init)

      loc = loc_layer(h)
      scale = scale_layer(h)
      scale = jax.nn.softplus(scale) + min_scale

      return TanhNormalParams(loc=loc, scale=scale)

    value_network = hk.Sequential([
        utils.batch_concat,
        hk.nets.MLP(value_layer_sizes, activate_final=True),
        hk.Linear(1),
        lambda x: jnp.squeeze(x, axis=-1)
    ])

    policy_output = _policy_network(inputs)
    value = value_network(inputs)
    return (policy_output, value)

  # Transform into pure functions.
  forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

  dummy_obs = utils.zeros_like(environment_spec.observations)
  dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.
  network = networks_lib.FeedForwardNetwork(
      lambda rng: forward_fn.init(rng, dummy_obs), forward_fn.apply)

  # Create PPONetworks to add functionality required by the agent.

  if not use_tanh_gaussian_policy:
    return make_mvn_diag_ppo_networks(network)

  return make_tanh_normal_ppo_networks(network)
