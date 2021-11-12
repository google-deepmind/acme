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

"""ActorCore interface definition."""

import dataclasses
from typing import Callable, Generic, Mapping, Tuple, TypeVar, Union

from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax.types import PRNGKey
import chex
import jax
import jax.numpy as jnp


NoneType = type(None)


# The state of the actor. This could include recurrent network state or any
# other state which needs to be propagated through the select_action calls.
State = TypeVar('State')
# The extras to be passed to the observe method.
Extras = TypeVar('Extras')
RecurrentState = TypeVar('RecurrentState')


@dataclasses.dataclass
class ActorCore(Generic[State, Extras]):
  """Pure functions that define the algorithm-specific actor functionality."""
  init: Callable[[PRNGKey], State]
  select_action: Callable[[
      networks_lib.Params, networks_lib.Observation, State
  ], Tuple[networks_lib.Action, State]]
  get_extras: Callable[[State], Extras]


# A simple feed forward policy which produces no extras and takes only an RNGKey
# as a state.
FeedForwardPolicy = Callable[
    [networks_lib.Params, PRNGKey, networks_lib.Observation],
    networks_lib.Action]

FeedForwardPolicyWithExtra = Callable[
    [networks_lib.Params, PRNGKey, networks_lib.Observation],
    Tuple[networks_lib.Action, types.NestedArray]]

RecurrentPolicy = Callable[[
    networks_lib.Params, PRNGKey, networks_lib
    .Observation, RecurrentState
], Tuple[networks_lib.Action, RecurrentState]]

Policy = Union[FeedForwardPolicy, FeedForwardPolicyWithExtra, RecurrentPolicy]


def batched_feed_forward_to_actor_core(
    policy: FeedForwardPolicy
) -> ActorCore[PRNGKey, NoneType]:
  """A convenience adaptor from FeedForwardPolicy to ActorCore."""

  def select_action(params: networks_lib.Params,
                    observation: networks_lib.Observation,
                    state: PRNGKey):
    rng = state
    rng1, rng2 = jax.random.split(rng)
    observation = utils.add_batch_dim(observation)
    action = utils.squeeze_batch_dim(policy(params, rng1, observation))
    return action, rng2

  def init(rng: PRNGKey) -> PRNGKey:
    return rng

  def get_extras(unused_rng: PRNGKey) -> NoneType:
    return None
  return ActorCore(init=init, select_action=select_action,
                   get_extras=get_extras)


@chex.dataclass(frozen=True, mappable_dataclass=False)
class SimpleActorCoreStateWithExtras:
  rng: PRNGKey
  extras: Mapping[str, jnp.ndarray]


def batched_feed_forward_with_extras_to_actor_core(
    policy: FeedForwardPolicyWithExtra
) -> ActorCore[SimpleActorCoreStateWithExtras, Mapping[str, jnp.ndarray]]:
  """A convenience adaptor from FeedForwardPolicy to ActorCore."""

  def select_action(params: networks_lib.Params,
                    observation: networks_lib.Observation,
                    state: SimpleActorCoreStateWithExtras):
    rng = state.rng
    rng1, rng2 = jax.random.split(rng)
    observation = utils.add_batch_dim(observation)
    action, extras = utils.squeeze_batch_dim(policy(params, rng1, observation))
    return action, SimpleActorCoreStateWithExtras(rng2, extras)

  def init(rng: PRNGKey) -> SimpleActorCoreStateWithExtras:
    return SimpleActorCoreStateWithExtras(rng, {})

  def get_extras(
      state: SimpleActorCoreStateWithExtras) -> Mapping[str, jnp.ndarray]:
    return state.extras
  return ActorCore(init=init, select_action=select_action,
                   get_extras=get_extras)


@chex.dataclass(frozen=True, mappable_dataclass=False)
class SimpleActorCoreRecurrentState(Generic[RecurrentState]):
  rng: PRNGKey
  recurrent_state: RecurrentState


def batched_recurrent_to_actor_core(
    recurrent_policy: RecurrentPolicy, initial_core_state: RecurrentState
) -> ActorCore[SimpleActorCoreRecurrentState[RecurrentState], Mapping[
    str, jnp.ndarray]]:
  """Returns ActorCore for a recurrent policy."""
  def select_action(params: networks_lib.Params,
                    observation: networks_lib.Observation,
                    state: SimpleActorCoreRecurrentState[RecurrentState]):
    # TODO(b/161332815): Make JAX Actor work with batched or unbatched inputs.
    rng = state.rng
    rng, policy_rng = jax.random.split(rng)
    observation = utils.add_batch_dim(observation)
    recurrent_state = utils.add_batch_dim(state.recurrent_state)
    action, new_recurrent_state = utils.squeeze_batch_dim(recurrent_policy(
        params, policy_rng, observation, recurrent_state))
    return action, SimpleActorCoreRecurrentState(rng, new_recurrent_state)

  initial_core_state = utils.squeeze_batch_dim(initial_core_state)
  def init(rng: PRNGKey) -> SimpleActorCoreRecurrentState[RecurrentState]:
    return SimpleActorCoreRecurrentState(rng, initial_core_state)

  def get_extras(
      state: SimpleActorCoreRecurrentState[RecurrentState]
  ) -> Mapping[str, jnp.ndarray]:
    return {'core_state': state.recurrent_state}

  return ActorCore(init=init, select_action=select_action,
                   get_extras=get_extras)
