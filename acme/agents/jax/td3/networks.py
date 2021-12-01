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

"""TD3 networks definition."""
import dataclasses
from typing import Callable, Sequence

from acme import specs
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass
class TD3Networks:
  """Network and pure functions for the TD3 agent."""
  policy_network: networks_lib.FeedForwardNetwork
  critic_network: networks_lib.FeedForwardNetwork
  twin_critic_network: networks_lib.FeedForwardNetwork
  add_policy_noise: Callable[[types.NestedArray, networks_lib.PRNGKey,
                              float, float], types.NestedArray]


def get_default_behavior_policy(networks: TD3Networks,
                                action_specs: specs.BoundedArray,
                                sigma: float):
  """Selects action according to the policy."""
  def behavior_policy(params: networks_lib.Params, key: networks_lib.PRNGKey,
                      observation: types.NestedArray):
    action = networks.policy_network.apply(params, observation)
    noise = jax.random.normal(key, shape=action.shape) * sigma
    noisy_action = jnp.clip(action + noise,
                            action_specs.minimum, action_specs.maximum)
    return noisy_action
  return behavior_policy


def make_networks(
    spec: specs.EnvironmentSpec,
    hidden_layer_sizes: Sequence[int] = (256, 256)) -> TD3Networks:
  """Creates networks used by the agent.

  The networks used are based on LayerNormMLP, which is different than the
  MLP with relu activation described in TD3 (which empirically performs worse).

  Args:
    spec: Environment specs
    hidden_layer_sizes: list of sizes of hidden layers in actor/critic networks

  Returns:
    network: TD3Networks
  """

  action_specs = spec.actions
  num_dimensions = np.prod(action_specs.shape, dtype=int)

  def add_policy_noise(action: types.NestedArray,
                       key: networks_lib.PRNGKey,
                       target_sigma: float,
                       noise_clip: float) -> types.NestedArray:
    """Adds action noise to bootstrapped Q-value estimate in critic loss."""
    noise = jax.random.normal(key=key, shape=action_specs.shape) * target_sigma
    noise = jnp.clip(noise, -noise_clip, noise_clip)
    return jnp.clip(action + noise, action_specs.minimum, action_specs.maximum)

  def _actor_fn(obs: types.NestedArray) -> types.NestedArray:
    network = hk.Sequential([
        networks_lib.LayerNormMLP(hidden_layer_sizes,
                                  activate_final=True),
        networks_lib.NearZeroInitializedLinear(num_dimensions),
        networks_lib.TanhToSpec(spec.actions),
    ])
    return network(obs)

  def _critic_fn(obs: types.NestedArray,
                 action: types.NestedArray) -> types.NestedArray:
    network1 = hk.Sequential([
        networks_lib.LayerNormMLP(list(hidden_layer_sizes) + [1]),
    ])
    input_ = jnp.concatenate([obs, action], axis=-1)
    value = network1(input_)
    return jnp.squeeze(value)

  policy = hk.without_apply_rng(hk.transform(_actor_fn))
  critic = hk.without_apply_rng(hk.transform(_critic_fn))

  # Create dummy observations and actions to create network parameters.
  dummy_action = utils.zeros_like(spec.actions)
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_action = utils.add_batch_dim(dummy_action)
  dummy_obs = utils.add_batch_dim(dummy_obs)

  network = TD3Networks(
      policy_network=networks_lib.FeedForwardNetwork(
          lambda key: policy.init(key, dummy_obs), policy.apply),
      critic_network=networks_lib.FeedForwardNetwork(
          lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply),
      twin_critic_network=networks_lib.FeedForwardNetwork(
          lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply),
      add_policy_noise=add_policy_noise)

  return network
