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

"""Acting logic for the MPO agent."""

from typing import Mapping, NamedTuple, Tuple, Union

from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax.mpo import networks
from acme.agents.jax.mpo import types
from acme.jax import types as jax_types
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class ActorState(NamedTuple):
  key: jax_types.PRNGKey
  core_state: hk.LSTMState
  prev_core_state: hk.LSTMState
  log_prob: Union[jnp.ndarray, Tuple[()]] = ()


def make_actor_core(mpo_networks: networks.MPONetworks,
                    stochastic: bool = True,
                    store_core_state: bool = False,
                    store_log_prob: bool = True) -> actor_core_lib.ActorCore:
  """Returns a MPO ActorCore from the MPONetworks."""

  def init(key: jax_types.PRNGKey) -> ActorState:
    next_key, key = jax.random.split(key, 2)
    batch_size = None
    params_initial_state = mpo_networks.torso.initial_state_fn_init(
        key, batch_size)
    core_state = mpo_networks.torso.initial_state_fn(params_initial_state,
                                                     batch_size)
    return ActorState(
        key=next_key,
        core_state=core_state,
        prev_core_state=core_state,
        log_prob=np.zeros(shape=(), dtype=np.float32) if store_log_prob else ())

  def select_action(params: networks.MPONetworkParams,
                    observations: types.Observation,
                    state: ActorState) -> Tuple[types.Action, ActorState]:

    next_key, key = jax.random.split(state.key, 2)

    # Embed observations and apply stateful core (e.g. recurrent, transformer).
    embeddings, core_state = mpo_networks.torso.apply(params.torso,
                                                      observations,
                                                      state.core_state)

    # Get the action distribution for these observations.
    policy = mpo_networks.policy_head_apply(params, embeddings)
    actions = policy.sample(seed=key) if stochastic else policy.mode()

    return actions, ActorState(
        key=next_key,
        core_state=core_state,
        prev_core_state=state.core_state,
        # Compute log-probabilities for use in off-policy correction schemes.
        log_prob=policy.log_prob(actions) if store_log_prob else ())

  def get_extras(state: ActorState) -> Mapping[str, jnp.ndarray]:
    extras = {}

    if store_core_state:
      extras['core_state'] = state.prev_core_state

    if store_log_prob:
      extras['log_prob'] = state.log_prob

    return extras  # pytype: disable=bad-return-type  # jax-ndarray

  return actor_core_lib.ActorCore(
      init=init, select_action=select_action, get_extras=get_extras)
