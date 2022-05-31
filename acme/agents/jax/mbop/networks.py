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

"""MBOP networks definitions."""

import dataclasses
from typing import Any, Tuple

from acme import specs
from acme.jax import networks
from acme.jax import utils
import haiku as hk
import jax.numpy as jnp
import numpy as np

# The term network is used in a general sense, e.g. for the CRR policy prior, it
# will be a dataclass that encapsulates the networks used by the CRR (learner).
WorldModelNetwork = Any
PolicyPriorNetwork = Any
NStepReturnNetwork = Any


@dataclasses.dataclass
class MBOPNetworks:
  """Container class to hold MBOP networks."""
  world_model_network: WorldModelNetwork
  policy_prior_network: PolicyPriorNetwork
  n_step_return_network: NStepReturnNetwork


def make_network_from_module(
    module: hk.Transformed,
    spec: specs.EnvironmentSpec) -> networks.FeedForwardNetwork:
  """Creates a network with dummy init arguments using the specified module.

  Args:
    module: Module that expects one batch axis and one features axis for its
      inputs.
    spec: EnvironmentSpec shapes to derive dummy inputs.

  Returns:
    FeedForwardNetwork whose `init` method only takes a random key, and `apply`
    takes an observation and action and produces an output.
  """
  dummy_obs = utils.add_batch_dim(utils.zeros_like(spec.observations))
  dummy_action = utils.add_batch_dim(utils.zeros_like(spec.actions))
  return networks.FeedForwardNetwork(
      lambda key: module.init(key, dummy_obs, dummy_action), module.apply)


def make_world_model_network(
    spec: specs.EnvironmentSpec, hidden_layer_sizes: Tuple[int, ...] = (64, 64)
) -> networks.FeedForwardNetwork:
  """Creates a world model network used by the agent."""

  observation_size = np.prod(spec.observations.shape, dtype=int)

  def _world_model_fn(observation_t, action_t, is_training=False, key=None):
    # is_training and key allows to defined train/test dependant modules
    # like dropout.
    del is_training
    del key
    network = hk.nets.MLP(hidden_layer_sizes + (observation_size + 1,))
    # World model returns both an observation and a reward.
    observation_tp1, reward_t = jnp.split(
        network(jnp.concatenate([observation_t, action_t], axis=-1)),
        [observation_size],
        axis=-1)
    return observation_tp1, reward_t

  world_model = hk.without_apply_rng(hk.transform(_world_model_fn))
  return make_network_from_module(world_model, spec)


def make_policy_prior_network(
    spec: specs.EnvironmentSpec, hidden_layer_sizes: Tuple[int, ...] = (64, 64)
) -> networks.FeedForwardNetwork:
  """Creates a policy prior network used by the agent."""

  action_size = np.prod(spec.actions.shape, dtype=int)

  def _policy_prior_fn(observation_t, action_tm1, is_training=False, key=None):
    # is_training and key allows to defined train/test dependant modules
    # like dropout.
    del is_training
    del key
    network = hk.nets.MLP(hidden_layer_sizes + (action_size,))
    # Policy prior returns an action.
    return network(jnp.concatenate([observation_t, action_tm1], axis=-1))

  policy_prior = hk.without_apply_rng(hk.transform(_policy_prior_fn))
  return make_network_from_module(policy_prior, spec)


def make_n_step_return_network(
    spec: specs.EnvironmentSpec, hidden_layer_sizes: Tuple[int, ...] = (64, 64)
) -> networks.FeedForwardNetwork:
  """Creates an N-step return network used by the agent."""

  def _n_step_return_fn(observation_t, action_t, is_training=False, key=None):
    # is_training and key allows to defined train/test dependant modules
    # like dropout.
    del is_training
    del key
    network = hk.nets.MLP(hidden_layer_sizes + (1,))
    return network(jnp.concatenate([observation_t, action_t], axis=-1))

  n_step_return = hk.without_apply_rng(hk.transform(_n_step_return_fn))
  return make_network_from_module(n_step_return, spec)


def make_networks(
    spec: specs.EnvironmentSpec,
    hidden_layer_sizes: Tuple[int, ...] = (64, 64),
) -> MBOPNetworks:
  """Creates networks used by the agent."""
  world_model_network = make_world_model_network(
      spec, hidden_layer_sizes=hidden_layer_sizes)
  policy_prior_network = make_policy_prior_network(
      spec, hidden_layer_sizes=hidden_layer_sizes)
  n_step_return_network = make_n_step_return_network(
      spec, hidden_layer_sizes=hidden_layer_sizes)

  return MBOPNetworks(
      world_model_network=world_model_network,
      policy_prior_network=policy_prior_network,
      n_step_return_network=n_step_return_network)
