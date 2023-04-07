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

"""DQN actor helpers."""

from typing import Callable, Sequence

import chex
import jax
import jax.numpy as jnp

from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax.dqn import networks as dqn_networks
from acme.jax import networks as networks_lib
from acme.jax import utils

Epsilon = float
EpsilonPolicy = Callable[
    [networks_lib.Params, networks_lib.PRNGKey, networks_lib.Observation, Epsilon],
    networks_lib.Action,
]


@chex.dataclass(frozen=True, mappable_dataclass=False)
class EpsilonActorState:
    rng: networks_lib.PRNGKey
    epsilon: jnp.ndarray


DQNPolicy = actor_core_lib.ActorCore[EpsilonActorState, None]


def alternating_epsilons_actor_core(
    policy_network: EpsilonPolicy, epsilons: Sequence[float]
) -> DQNPolicy:
    """Returns actor components for alternating epsilon exploration.

  Args:
    policy_network: A feedforward action selecting function.
    epsilons: epsilons to alternate per-episode for epsilon-greedy exploration.

  Returns:
    A feedforward policy.
  """
    epsilons = jnp.array(epsilons)

    def apply_and_sample(
        params: networks_lib.Params,
        observation: networks_lib.Observation,
        state: EpsilonActorState,
    ):
        random_key, key = jax.random.split(state.rng)
        actions = policy_network(
            params, key, observation, state.epsilon
        )  # pytype: disable=wrong-arg-types  # jax-ndarray
        return (
            actions.astype(jnp.int32),
            EpsilonActorState(rng=random_key, epsilon=state.epsilon),
        )

    def policy_init(random_key: networks_lib.PRNGKey):
        random_key, key = jax.random.split(random_key)
        epsilon = jax.random.choice(key, epsilons)
        return EpsilonActorState(rng=random_key, epsilon=epsilon)

    return actor_core_lib.ActorCore(
        init=policy_init, select_action=apply_and_sample, get_extras=lambda _: None
    )


def behavior_policy(networks: dqn_networks.DQNNetworks) -> EpsilonPolicy:
    """A policy with parameterized epsilon-greedy exploration."""

    def apply_and_sample(
        params: networks_lib.Params,
        key: networks_lib.PRNGKey,
        observation: networks_lib.Observation,
        epsilon: Epsilon,
    ) -> networks_lib.Action:
        # TODO(b/161332815): Make JAX Actor work with batched or unbatched inputs.
        observation = utils.add_batch_dim(observation)
        action_values = networks.policy_network.apply(
            params, observation, is_training=False
        )
        action_values = utils.squeeze_batch_dim(action_values)
        return networks.sample_fn(action_values, key, epsilon)

    return apply_and_sample


def default_behavior_policy(
    networks: dqn_networks.DQNNetworks, epsilon: Epsilon
) -> EpsilonPolicy:
    """A policy with a fixed-epsilon epsilon-greedy exploration.

  DEPRECATED: use behavior_policy instead.
  Args:
    networks: DQN networks
    epsilon: sampling parameter that overrides the one in EpsilonPolicy
  Returns:
    epsilon-greedy behavior policy with fixed epsilon
  """
    # TODO(lukstafi): remove this function and migrate its users.

    def apply_and_sample(
        params: networks_lib.Params,
        key: networks_lib.PRNGKey,
        observation: networks_lib.Observation,
        _: Epsilon,
    ) -> networks_lib.Action:
        # TODO(b/161332815): Make JAX Actor work with batched or unbatched inputs.
        observation = utils.add_batch_dim(observation)
        action_values = networks.policy_network.apply(
            params, observation, is_training=False
        )
        action_values = utils.squeeze_batch_dim(action_values)
        return networks.sample_fn(action_values, key, epsilon)

    return apply_and_sample
