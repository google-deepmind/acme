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

from typing import Callable, Generic, Optional, Tuple, TypeVar, Union

from acme import adders
from acme import core
from acme import types
from acme.agents.jax import actor_core
from acme.jax import networks as network_lib
from acme.jax import utils
from acme.jax import variable_utils
import dm_env
import jax

# Useful type aliases.
RecurrentState = TypeVar('RecurrentState')

# Signatures for functions that sample from parameterised stochastic policies.
FeedForwardPolicy = Callable[
    [network_lib.Params, network_lib.PRNGKey, network_lib.Observation],
    Union[network_lib.Action, Tuple[network_lib.Action, types.NestedArray]]]
RecurrentPolicy = Callable[[
    network_lib.Params, network_lib.PRNGKey, network_lib
    .Observation, RecurrentState
], Tuple[Union[network_lib.Action, Tuple[network_lib.Action,
                                         types.NestedArray]], RecurrentState]]


class GenericActor(core.Actor, Generic[actor_core.State, actor_core.Extras]):
  """A generic actor implemented on top of ActorCore.

  An actor based on a policy which takes observations and outputs actions. It
  also adds experiences to replay and updates the actor weights from the policy
  on the learner.
  """

  def __init__(
      self,
      actor: actor_core.ActorCore[actor_core.State, actor_core.Extras],
      random_key: network_lib.PRNGKey,
      variable_client: variable_utils.VariableClient,
      adder: Optional[adders.Adder] = None,
      backend: Optional[str] = 'cpu',
  ):
    """Initializes a feed forward actor.

    Args:
      actor: actor core.
      random_key: Random key.
      variable_client: The variable client to get policy parameters from.
      adder: An adder to add experiences to.
      backend: Which backend to use for running the policy.
    """
    self._random_key = random_key

    self._init = jax.jit(actor.init)
    self._policy = jax.jit(actor.select_action, backend=backend)
    self._get_extras = actor.get_extras

    self._adder = adder
    self._state = None
    self._client = variable_client

  def select_action(self,
                    observation: network_lib.Observation) -> types.NestedArray:
    action, self._state = self._policy(self._client.params, observation,
                                       self._state)
    return utils.to_numpy(action)

  def observe_first(self, timestep: dm_env.TimeStep):
    self._random_key, key = jax.random.split(self._random_key)
    self._state = self._init(key)
    if self._adder:
      self._adder.add_first(timestep)

  def observe(self, action: network_lib.Action, next_timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add(
          action, next_timestep, extras=self._get_extras(self._state))

  def update(self, wait: bool = False):
    self._client.update(wait)


# TODO(raveman): Migrate all users of FeedForwardActor to GenericActor and
# remove this class.
class FeedForwardActor(core.Actor):
  """A simple feed-forward actor implemented in JAX.

  An actor based on a policy which takes observations and outputs actions. It
  also adds experiences to replay and updates the actor weights from the policy
  on the learner.
  """

  def __init__(
      self,
      policy: FeedForwardPolicy,
      random_key: network_lib.PRNGKey,
      variable_client: variable_utils.VariableClient,
      adder: Optional[adders.Adder] = None,
      has_extras: bool = False,
      backend: Optional[str] = 'cpu',
  ):
    """Initializes a feed forward actor.

    Args:
      policy: A policy network taking observation and returning an action, if
        `has_extras=False`, and returning an (action, extras) tuple if
        `has_extras=True`.
      random_key: Random key.
      variable_client: The variable client to get policy parameters from.
      adder: An adder to add experiences to.
      has_extras: Flag indicating whether the policy returns extra information
        (e.g. q-values) in addition to an action.
      backend: Which backend to use for running the policy.
    """
    self._random_key = random_key
    self._has_extras = has_extras
    self._extras: types.NestedArray = ()

    # Adding batch dimension inside jit is much more efficient than outside.
    def batched_policy(
        params: network_lib.Params, key: network_lib.PRNGKey,
        observation: network_lib.Observation
    ) -> Tuple[Union[network_lib.Action, Tuple[
        network_lib.Action, types.NestedArray]], network_lib.PRNGKey]:
      # TODO(b/161332815): Make JAX Actor work with batched or unbatched inputs.
      key, key2 = jax.random.split(key)
      observation = utils.add_batch_dim(observation)
      output = policy(params, key2, observation)
      return utils.squeeze_batch_dim(output), key

    self._policy = jax.jit(batched_policy, backend=backend)

    self._adder = adder
    self._client = variable_client

  def select_action(self,
                    observation: network_lib.Observation) -> types.NestedArray:
    result, self._random_key = self._policy(self._client.params,
                                            self._random_key, observation)
    if self._has_extras:
      action, self._extras = result
    else:
      action = result
    return utils.to_numpy(action)

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

  def observe(self, action: network_lib.Action, next_timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add(action, next_timestep, extras=self._extras)

  def update(self, wait: bool = False):
    self._client.update(wait)


# TODO(raveman): Migrate all users of RecurrentActor to GenericActor and
# remove this class.
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
      random_key: network_lib.PRNGKey,
      initial_core_state: RecurrentState,
      variable_client: variable_utils.VariableClient,
      adder: Optional[adders.Adder] = None,
      has_extras: bool = False,
      backend: Optional[str] = 'cpu',
  ):
    """Initializes a recurrent actor.

    Args:
      recurrent_policy: A recurrent policy network taking observation and state
        and returning an (action, state) tuple, if `has_extras=False`, and
        returning an ((action, extras), state) tuple if `has_extras=True`. In
        the latter case, `extras` must be a tuple.
      random_key: Random key.
      initial_core_state: Initial state of the recurrent policy.
      variable_client: The variable client to get policy parameters from.
      adder: An adder to add experiences to. The `extras` of the adder holds the
        state of the recurrent policy. If `has_extras=True` then the `extras`
        part returned from the recurrent policy is appended to the state before
        added to the adder.
      has_extras: Flag indicating whether the recurrent policy returns extra
        information (e.g. q-values) in addition to an action.
      backend: Which backend to use for running the policy.
    """
    self._random_key = random_key
    self._has_extras = has_extras
    self._extras: types.NestedArray = ()

    # Adding batch dimension inside jit is much more efficient than outside.
    def batched_recurrent_policy(
        params: network_lib.Params, key: network_lib.PRNGKey,
        observation: network_lib.Observation, core_state: RecurrentState
    ) -> Tuple[Union[network_lib.Action, Tuple[
        network_lib.Action, types.NestedArray]], RecurrentState,
               network_lib.PRNGKey]:
      # TODO(b/161332815): Make JAX Actor work with batched or unbatched inputs.
      observation = utils.add_batch_dim(observation)
      key, key2 = jax.random.split(key)
      output, new_state = recurrent_policy(params, key2, observation,
                                           core_state)
      return output, new_state, key

    self._recurrent_policy = jax.jit(batched_recurrent_policy, backend=backend)

    self._initial_state = self._prev_state = self._state = initial_core_state
    self._adder = adder
    self._client = variable_client

  def select_action(self,
                    observation: network_lib.Observation) -> network_lib.Action:
    result, new_state, self._random_key = self._recurrent_policy(
        self._client.params,
        key=self._random_key,
        observation=observation,
        core_state=self._state)
    self._prev_state = self._state  # Keep previous state to save in replay.
    self._state = new_state  # Keep new state for next policy call.

    if self._has_extras:
      action, extras = result
      self._extras = utils.to_numpy_squeeze(extras)  # Keep to save in replay.
    else:
      action = result
    return utils.to_numpy_squeeze(action)

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)
    # Re-initialize state at beginning of new episode.
    self._state = self._initial_state

  def observe(self, action: network_lib.Action, next_timestep: dm_env.TimeStep):
    if self._adder:
      # Convert state to numpy array and pack it in a dictionary.
      extras = {'core_state': utils.to_numpy_squeeze(self._prev_state)}

      # Add core state to the extras dict or tuple, accordingly.
      if self._has_extras:
        if 'core_state' in self._extras:
          raise ValueError(
              'The policy network\'s extras dict already has a `core_state` '
              'but the actor is attempting to add its own `core_state`.')
        extras.update(self._extras)

      # Add the transition to the adder.
      self._adder.add(action, next_timestep, extras=extras)

  def update(self, wait: bool = False):
    self._client.update(wait)
