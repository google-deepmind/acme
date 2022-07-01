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

"""R2D2 actor."""

from typing import Callable, Generic, Mapping, Optional, Tuple

from acme import types
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax.r2d2 import config as r2d2_config
from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.jax import networks as networks_lib
import chex
import jax
import jax.numpy as jnp
import numpy as np
import rlax

Epsilon = float
R2D2Extras = Mapping[str, jnp.ndarray]
EpsilonRecurrentPolicy = Callable[[
    networks_lib.Params, networks_lib.PRNGKey, networks_lib
    .Observation, actor_core_lib.RecurrentState, Epsilon
], Tuple[networks_lib.Action, actor_core_lib.RecurrentState]]


@chex.dataclass(frozen=True, mappable_dataclass=False)
class R2D2ActorState(Generic[actor_core_lib.RecurrentState]):
  rng: networks_lib.PRNGKey
  epsilon: jnp.ndarray
  recurrent_state: actor_core_lib.RecurrentState


R2D2Policy = actor_core_lib.ActorCore[
    R2D2ActorState[actor_core_lib.RecurrentState], R2D2Extras]


def get_actor_core(
    networks: r2d2_networks.R2D2Networks,
    num_epsilons: Optional[int],
    evaluation_epsilon: Optional[float] = None,
) -> R2D2Policy:
  """Returns ActorCore for R2D2."""

  if (not num_epsilons and evaluation_epsilon is None) or (num_epsilons and
                                                           evaluation_epsilon):
    raise ValueError(
        'Exactly one of `num_epsilons` or `evaluation_epsilon` must be '
        f'specified. Received num_epsilon={num_epsilons} and '
        f'evaluation_epsilon={evaluation_epsilon}.')

  def select_action(params: networks_lib.Params,
                    observation: networks_lib.Observation,
                    state: R2D2ActorState[actor_core_lib.RecurrentState]):
    rng, policy_rng = jax.random.split(state.rng)

    q_values, recurrent_state = networks.forward.apply(params, policy_rng,
                                                       observation,
                                                       state.recurrent_state)
    action = rlax.epsilon_greedy(state.epsilon).sample(policy_rng, q_values)

    return action, R2D2ActorState(rng, state.epsilon, recurrent_state)

  def init(
      rng: networks_lib.PRNGKey
  ) -> R2D2ActorState[actor_core_lib.RecurrentState]:
    rng, epsilon_rng, state_rng = jax.random.split(rng, 3)
    if num_epsilons:
      epsilon = jax.random.choice(epsilon_rng,
                                  np.logspace(1, 8, num_epsilons, base=0.4))
    else:
      epsilon = evaluation_epsilon
    initial_core_state = networks.initial_state.apply(None, state_rng, None)
    return R2D2ActorState(rng, epsilon, initial_core_state)

  def get_extras(
      state: R2D2ActorState[actor_core_lib.RecurrentState]) -> R2D2Extras:
    return {'core_state': state.recurrent_state}

  return actor_core_lib.ActorCore(init=init, select_action=select_action,
                                  get_extras=get_extras)


# TODO(bshahr): Deprecate this in favour of R2D2Builder.make_policy.
def make_behavior_policy(networks: r2d2_networks.R2D2Networks,
                         config: r2d2_config.R2D2Config,
                         evaluation: bool = False) -> EpsilonRecurrentPolicy:
  """Selects action according to the policy."""

  def behavior_policy(params: networks_lib.Params, key: networks_lib.PRNGKey,
                      observation: types.NestedArray,
                      core_state: types.NestedArray, epsilon: float):
    q_values, core_state = networks.forward.apply(params, key, observation,
                                                  core_state)
    epsilon = config.evaluation_epsilon if evaluation else epsilon
    return rlax.epsilon_greedy(epsilon).sample(key, q_values), core_state

  return behavior_policy
