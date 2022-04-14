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

"""Loss (weight) functions for CRR."""

from typing import Callable

from acme import types
from acme.agents.jax.crr.networks import CRRNetworks
from acme.jax import networks as networks_lib
import jax.numpy as jnp

PolicyLossCoeff = Callable[[
    CRRNetworks,
    networks_lib.Params,
    networks_lib.Params,
    types.Transition,
    networks_lib.PRNGKey,
], jnp.ndarray]


def _compute_advantage(networks: CRRNetworks,
                       policy_params: networks_lib.Params,
                       critic_params: networks_lib.Params,
                       transition: types.Transition,
                       key: networks_lib.PRNGKey,
                       num_action_samples: int = 4) -> jnp.ndarray:
  """Returns the advantage for the transition."""
  # Sample count actions.
  replicated_observation = jnp.broadcast_to(transition.observation,
                                            (num_action_samples,) +
                                            transition.observation.shape)
  dist_params = networks.policy_network.apply(policy_params,
                                              replicated_observation)
  actions = networks.sample(dist_params, key)
  # Compute the state-action values for the sampled actions.
  q_actions = networks.critic_network.apply(critic_params,
                                            replicated_observation, actions)
  # Take the mean as the state-value estimate. It is also possible to take the
  # maximum, aka CRR(max); see table 1 in CRR paper.
  q_estimate = jnp.mean(q_actions, axis=0)
  # Compute the advantage.
  q = networks.critic_network.apply(critic_params, transition.observation,
                                    transition.action)
  return q - q_estimate


def policy_loss_coeff_advantage_exp(
    networks: CRRNetworks,
    policy_params: networks_lib.Params,
    critic_params: networks_lib.Params,
    transition: types.Transition,
    key: networks_lib.PRNGKey,
    num_action_samples: int = 4,
    beta: float = 1.0,
    ratio_upper_bound: float = 20.0) -> jnp.ndarray:
  """Exponential advantage weigting; see equation (4) in CRR paper."""
  advantage = _compute_advantage(networks, policy_params, critic_params,
                                 transition, key, num_action_samples)
  return jnp.minimum(jnp.exp(advantage / beta), ratio_upper_bound)


def policy_loss_coeff_advantage_indicator(
    networks: CRRNetworks,
    policy_params: networks_lib.Params,
    critic_params: networks_lib.Params,
    transition: types.Transition,
    key: networks_lib.PRNGKey,
    num_action_samples: int = 4) -> jnp.ndarray:
  """Indicator advantage weighting; see equation (3) in CRR paper."""
  advantage = _compute_advantage(networks, policy_params, critic_params,
                                 transition, key, num_action_samples)
  return jnp.heaviside(advantage, 0.)


def policy_loss_coeff_constant(networks: CRRNetworks,
                               policy_params: networks_lib.Params,
                               critic_params: networks_lib.Params,
                               transition: types.Transition,
                               key: networks_lib.PRNGKey,
                               value: float = 1.0) -> jnp.ndarray:
  """Constant weights."""
  del networks
  del policy_params
  del critic_params
  del transition
  del key
  return value
