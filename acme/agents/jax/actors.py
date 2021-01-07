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

"""Simple JAX actors."""

from typing import Callable, Optional, Tuple, TypeVar, Union

from acme import adders
from acme import core
from acme import types
from acme.jax import utils
from acme.jax import variable_utils

import dm_env
import haiku as hk
import jax
import jax.numpy as jnp

# Useful type aliases.
RNGKey = jnp.ndarray
Observation = types.NestedArray
Action = types.NestedArray
RecurrentState = TypeVar('RecurrentState')

# Signatures for functions that sample from parameterised stochastic policies.
FeedForwardPolicy = Callable[[hk.Params, RNGKey, Observation],
                             Union[Action, Tuple[Action, types.NestedArray]]]
RecurrentPolicy = Callable[[hk.Params, RNGKey, Observation, RecurrentState],
                           Tuple[Action, RecurrentState]]


class FeedForwardActor(core.Actor):
  """A simple feed-forward actor implemented in JAX.

  An actor based on a policy which takes observations and outputs actions. It
  also adds experiences to replay and updates the actor weights from the policy
  on the learner.
  """

  def __init__(
      self,
      policy: FeedForwardPolicy,
      rng: hk.PRNGSequence,
      variable_client: variable_utils.VariableClient,
      adder: Optional[adders.Adder] = None,
      has_extras: bool = False,
  ):
    """Initializes a feed forward actor.

    Args:
      policy: A policy network taking observation and returning an action, if
        `has_extras=False`, and returning an (action, extras) tuple if
        `has_extras=True`.
      rng: Random key generator.
      variable_client: The variable client to get policy parameters from.
      adder: An adder to add experiences to.
      has_extras: Flag indicating whether the policy returns extra
        information (e.g. q-values) in addition to an action.
    """
    self._rng = rng
    self._has_extras = has_extras
    self._extras: types.NestedArray = ()

    # Adding batch dimension inside jit is much more efficient than outside.
    def batched_policy(params: hk.Params, key: RNGKey,
                       observation: Observation
                       ) ->Union[Action, Tuple[Action, types.NestedArray]]:
      # TODO(b/161332815): Make JAX Actor work with batched or unbatched inputs.
      observation = utils.add_batch_dim(observation)
      output = policy(params, key, observation)
      return utils.squeeze_batch_dim(output)
    self._policy = jax.jit(batched_policy, backend='cpu')

    self._adder = adder
    self._client = variable_client

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    key = next(self._rng)
    result = self._policy(self._client.params, key, observation)
    if self._has_extras:
      action, self._extras = result
    else:
      action = result
    return utils.to_numpy(action)

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add(action, next_timestep, extras=self._extras)

  def update(self, wait: bool = False):
    self._client.update(wait)


class RecurrentActor(core.Actor):
  """A recurrent actor in JAX.

  An actor based on a recurrent policy which takes observations and outputs
  actions, and keeps track of the recurrent state inside. It also adds
  experiences to replay and updates the actor weights from the policy on the
  learner.
  """

  def __init__(
      self,
      recurrent_policy: RecurrentPolicy,
      rng: hk.PRNGSequence,
      initial_core_state: RecurrentState,
      variable_client: variable_utils.VariableClient,
      adder: Optional[adders.Adder] = None,
      has_extras: bool = False,
  ):
    """Initializes a recurrent actor.

    Args:
      recurrent_policy: A recurrent policy network taking observation and state
        and returning an (action, state) tuple, if `has_extras=False`, and
        returning an ((action, extras), state) tuple if `has_extras=True`. In
        the latter case, `extras` must be a tuple.
      rng: Random key generator.
      initial_core_state: Initial state of the recurrent policy.
      variable_client: The variable client to get policy parameters from.
      adder: An adder to add experiences to. The `extras` of the adder holds the
        state of the recurrent policy. If `has_extras=True` then the `extras`
        part returned from the recurrent policy is appended to the state before
        added to the adder.
      has_extras: Flag indicating whether the recurrent policy returns extra
        information (e.g. q-values) in addition to an action.
    """
    self._rng = rng
    self._has_extras = has_extras
    self._extras: types.NestedArray = ()

    # Adding batch dimension inside jit is much more efficient than outside.
    def batched_recurrent_policy(params, key, observation, core_state):
      # TODO(b/161332815): Make JAX Actor work with batched or unbatched inputs.
      observation = utils.add_batch_dim(observation)
      output, new_state = recurrent_policy(params, key, observation, core_state)
      return output, new_state
    self._recurrent_policy = jax.jit(batched_recurrent_policy, backend='cpu')

    self._initial_state = self._prev_state = self._state = initial_core_state
    self._adder = adder
    self._client = variable_client

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    result, new_state = self._recurrent_policy(
        self._client.params,
        key=next(self._rng),
        observation=observation,
        core_state=self._state)
    self._prev_state = self._state  # Keep previous state to save in replay.
    self._state = new_state  # Keep new state for next policy call.

    if self._has_extras:
      action, extras = result
      self._extras = utils.to_numpy_squeeze(extras)   # Keep to save in replay.
    else:
      action = result
    return utils.to_numpy_squeeze(action)

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)
    # Re-initialize state at beginning of new episode.
    self._state = self._initial_state

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    if self._adder:
      numpy_state = utils.to_numpy_squeeze(self._prev_state)
      self._adder.add(
          action, next_timestep, extras=(numpy_state,) + self._extras)

  def update(self, wait: bool = False):
    self._client.update(wait)
