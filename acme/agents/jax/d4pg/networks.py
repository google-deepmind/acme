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

"""D4PG networks definition."""

import dataclasses
from typing import Sequence

from acme import specs
from acme import types
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax.d4pg import config as d4pg_config
from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import jax.numpy as jnp
import numpy as np
import rlax


@dataclasses.dataclass
class D4PGNetworks:
  """Network and pure functions for the D4PG agent.."""
  policy_network: networks_lib.FeedForwardNetwork
  critic_network: networks_lib.FeedForwardNetwork


def get_default_behavior_policy(
    networks: D4PGNetworks,
    config: d4pg_config.D4PGConfig) -> actor_core_lib.FeedForwardPolicy:
  """Selects action according to the training policy."""
  def behavior_policy(params: networks_lib.Params, key: networks_lib.PRNGKey,
                      observation: types.NestedArray):
    action = networks.policy_network.apply(params, observation)
    if config.sigma != 0:
      action = rlax.add_gaussian_noise(key, action, config.sigma)
    return action

  return behavior_policy


def get_default_eval_policy(
    networks: D4PGNetworks) -> actor_core_lib.FeedForwardPolicy:
  """Selects action according to the training policy."""
  def behavior_policy(params: networks_lib.Params, key: networks_lib.PRNGKey,
                      observation: types.NestedArray):
    del key
    action = networks.policy_network.apply(params, observation)
    return action
  return behavior_policy


def make_networks(
    spec: specs.EnvironmentSpec,
    policy_layer_sizes: Sequence[int] = (300, 200),
    critic_layer_sizes: Sequence[int] = (400, 300),
    vmin: float = -150.,
    vmax: float = 150.,
    num_atoms: int = 51,
) -> D4PGNetworks:
  """Creates networks used by the agent."""

  action_spec = spec.actions

  num_dimensions = np.prod(action_spec.shape, dtype=int)
  critic_atoms = jnp.linspace(vmin, vmax, num_atoms)

  def _actor_fn(obs):
    network = hk.Sequential([
        utils.batch_concat,
        networks_lib.LayerNormMLP(policy_layer_sizes, activate_final=True),
        networks_lib.NearZeroInitializedLinear(num_dimensions),
        networks_lib.TanhToSpec(action_spec),
    ])
    return network(obs)

  def _critic_fn(obs, action):
    network = hk.Sequential([
        utils.batch_concat,
        networks_lib.LayerNormMLP(layer_sizes=[*critic_layer_sizes, num_atoms]),
    ])
    value = network([obs, action])
    return value, critic_atoms

  policy = hk.without_apply_rng(hk.transform(_actor_fn))
  critic = hk.without_apply_rng(hk.transform(_critic_fn))

  # Create dummy observations and actions to create network parameters.
  dummy_action = utils.zeros_like(spec.actions)
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_action = utils.add_batch_dim(dummy_action)
  dummy_obs = utils.add_batch_dim(dummy_obs)

  return D4PGNetworks(
      policy_network=networks_lib.FeedForwardNetwork(
          lambda rng: policy.init(rng, dummy_obs), policy.apply),
      critic_network=networks_lib.FeedForwardNetwork(
          lambda rng: critic.init(rng, dummy_obs, dummy_action), critic.apply))
