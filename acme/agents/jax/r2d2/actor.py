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

"""R2D2 actor."""

from typing import Generic, Mapping

from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.jax import networks as networks_lib
from acme.jax import utils
import chex
import jax
import jax.numpy as jnp


@chex.dataclass(frozen=True, mappable_dataclass=False)
class R2D2ActorState(Generic[actor_core_lib.RecurrentState]):
  rng: networks_lib.PRNGKey
  epsilon: jnp.ndarray
  recurrent_state: actor_core_lib.RecurrentState


def get_actor_core(
    recurrent_policy: r2d2_networks.EpsilonRecurrentPolicy[
        actor_core_lib.RecurrentState],
    initial_core_state: actor_core_lib.RecurrentState, num_epsilons: int
) -> actor_core_lib.ActorCore[R2D2ActorState[actor_core_lib.RecurrentState],
                              Mapping[str, jnp.ndarray]]:
  """Returns ActorCore for R2D2."""
  def select_action(params: networks_lib.Params,
                    observation: networks_lib.Observation,
                    state: R2D2ActorState[actor_core_lib.RecurrentState]):
    # TODO(b/161332815): Make JAX Actor work with batched or unbatched inputs.
    rng, policy_rng = jax.random.split(state.rng)
    observation = utils.add_batch_dim(observation)
    recurrent_state = utils.add_batch_dim(state.recurrent_state)
    action, new_recurrent_state = utils.squeeze_batch_dim(recurrent_policy(
        params, policy_rng, observation, recurrent_state, state.epsilon))
    return action, R2D2ActorState(rng, state.epsilon, new_recurrent_state)

  initial_core_state = utils.squeeze_batch_dim(initial_core_state)

  def init(
      rng: networks_lib.PRNGKey
  ) -> R2D2ActorState[actor_core_lib.RecurrentState]:
    rng, epsilon_rng = jax.random.split(rng)
    epsilon = jax.random.choice(epsilon_rng,
                                jnp.logspace(1, 8, num_epsilons, base=0.4))
    return R2D2ActorState(rng, epsilon, initial_core_state)

  def get_extras(
      state: R2D2ActorState[actor_core_lib.RecurrentState]
  ) -> Mapping[str, jnp.ndarray]:
    return {'core_state': state.recurrent_state}

  return actor_core_lib.ActorCore(init=init, select_action=select_action,
                                  get_extras=get_extras)
