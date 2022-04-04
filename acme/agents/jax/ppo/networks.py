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

"""PPO network definitions."""

import dataclasses
from typing import Any, Callable, Optional, Sequence

from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from acme.jax import utils

import haiku as hk
import jax.numpy as jnp
import numpy as np

EntropyFn = Callable[[Any], jnp.ndarray]


@dataclasses.dataclass
class PPONetworks:
  """Network and pure functions for the PPO agent.

  If 'network' returns tfd.Distribution, you can use make_ppo_networks() to
  create this object properly.
  If one is building this object manually, one has a freedom to make 'network'
  object return anything that is later being passed as input to
  log_prob/entropy/sample functions to perform the corresponding computations.
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

  def inference(params: networks_lib.Params, key: networks_lib.PRNGKey,
                observations: networks_lib.Observation):
    distribution, _ = ppo_networks.network.apply(params, observations)
    if evaluation and ppo_networks.sample_eval:
      actions = ppo_networks.sample_eval(distribution, key)
    else:
      actions = ppo_networks.sample(distribution, key)
    if evaluation:
      return actions, {}
    log_prob = ppo_networks.log_prob(distribution, actions)
    return actions, {'log_prob': log_prob}

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

  Args:
    network: a transformed Haiku network that takes in observations and returns
      the action distribution and value.

  Returns:
    A PPONetworks instance with pure functions wrapping the input network.
  """
  return PPONetworks(
      network=network,
      log_prob=lambda distribution, action: distribution.log_prob(action),
      entropy=lambda distribution: distribution.entropy(),
      sample=lambda distribution, key: distribution.sample(seed=key),
      sample_eval=lambda distribution, key: distribution.mode())


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
    layers.extend([
        hk.nets.MLP(hidden_layer_sizes, activate_final=True),
        networks_lib.CategoricalValueHead(num_values=num_actions)
    ])
    policy_value_network = hk.Sequential(layers)
    return policy_value_network(inputs)

  forward_fn = hk.without_apply_rng(hk.transform(forward_fn))
  dummy_obs = utils.zeros_like(environment_spec.observations)
  dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.
  network = networks_lib.FeedForwardNetwork(
      lambda rng: forward_fn.init(rng, dummy_obs), forward_fn.apply)
  # Create PPONetworks to add functionality required by the agent.
  return make_ppo_networks(network)


def make_continuous_networks(
    environment_spec: specs.EnvironmentSpec,
    policy_layer_sizes: Sequence[int] = (64, 64),
    value_layer_sizes: Sequence[int] = (64, 64),
) -> PPONetworks:
  """Creates PPONetworks to be used for continuous action environments."""

  # Get total number of action dimensions from action spec.
  num_dimensions = np.prod(environment_spec.actions.shape, dtype=int)

  def forward_fn(inputs):
    policy_network = hk.Sequential([
        utils.batch_concat,
        hk.nets.MLP(policy_layer_sizes, activate_final=True),
        # Note: we don't respect bounded action specs here and instead
        # rely on CanonicalSpecWrapper to clip actions accordingly.
        networks_lib.MultivariateNormalDiagHead(num_dimensions)
    ])
    value_network = hk.Sequential([
        utils.batch_concat,
        hk.nets.MLP(value_layer_sizes, activate_final=True),
        hk.Linear(1),
        lambda x: jnp.squeeze(x, axis=-1)
    ])

    action_distribution = policy_network(inputs)
    value = value_network(inputs)
    return (action_distribution, value)

  # Transform into pure functions.
  forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

  dummy_obs = utils.zeros_like(environment_spec.observations)
  dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.
  network = networks_lib.FeedForwardNetwork(
      lambda rng: forward_fn.init(rng, dummy_obs), forward_fn.apply)
  # Create PPONetworks to add functionality required by the agent.
  return make_ppo_networks(network)
