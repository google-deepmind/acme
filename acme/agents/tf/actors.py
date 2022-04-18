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

"""Generic actor implementation, using TensorFlow and Sonnet."""

from typing import Optional, Tuple

from acme import adders
from acme import core
from acme import types
# Internal imports.
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

import dm_env
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class FeedForwardActor(core.Actor):
  """A feed-forward actor.

  An actor based on a feed-forward policy which takes non-batched observations
  and outputs non-batched actions. It also allows adding experiences to replay
  and updating the weights from the policy on the learner.
  """

  def __init__(
      self,
      policy_network: snt.Module,
      adder: Optional[adders.Adder] = None,
      variable_client: Optional[tf2_variable_utils.VariableClient] = None,
  ):
    """Initializes the actor.

    Args:
      policy_network: the policy to run.
      adder: the adder object to which allows to add experiences to a
        dataset/replay buffer.
      variable_client: object which allows to copy weights from the learner copy
        of the policy to the actor copy (in case they are separate).
    """

    # Store these for later use.
    self._adder = adder
    self._variable_client = variable_client
    self._policy_network = policy_network

  @tf.function
  def _policy(self, observation: types.NestedTensor) -> types.NestedTensor:
    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_observation = tf2_utils.add_batch_dim(observation)

    # Compute the policy, conditioned on the observation.
    policy = self._policy_network(batched_observation)

    # Sample from the policy if it is stochastic.
    action = policy.sample() if isinstance(policy, tfd.Distribution) else policy

    return action

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    # Pass the observation through the policy network.
    action = self._policy(observation)

    # Return a numpy array with squeezed out batch dimension.
    return tf2_utils.to_numpy_squeeze(action)

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add(action, next_timestep)

  def update(self, wait: bool = False):
    if self._variable_client:
      self._variable_client.update(wait)


class RecurrentActor(core.Actor):
  """A recurrent actor.

  An actor based on a recurrent policy which takes non-batched observations and
  outputs non-batched actions, and keeps track of the recurrent state inside. It
  also allows adding experiences to replay and updating the weights from the
  policy on the learner.
  """

  def __init__(
      self,
      policy_network: snt.RNNCore,
      adder: Optional[adders.Adder] = None,
      variable_client: Optional[tf2_variable_utils.VariableClient] = None,
      store_recurrent_state: bool = True,
  ):
    """Initializes the actor.

    Args:
      policy_network: the (recurrent) policy to run.
      adder: the adder object to which allows to add experiences to a
        dataset/replay buffer.
      variable_client: object which allows to copy weights from the learner copy
        of the policy to the actor copy (in case they are separate).
      store_recurrent_state: Whether to pass the recurrent state to the adder.
    """
    # Store these for later use.
    self._adder = adder
    self._variable_client = variable_client
    self._network = policy_network
    self._state = None
    self._prev_state = None
    self._store_recurrent_state = store_recurrent_state

  @tf.function
  def _policy(
      self,
      observation: types.NestedTensor,
      state: types.NestedTensor,
  ) -> Tuple[types.NestedTensor, types.NestedTensor]:

    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_observation = tf2_utils.add_batch_dim(observation)

    # Compute the policy, conditioned on the observation.
    policy, new_state = self._network(batched_observation, state)

    # Sample from the policy if it is stochastic.
    action = policy.sample() if isinstance(policy, tfd.Distribution) else policy

    return action, new_state

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    # Initialize the RNN state if necessary.
    if self._state is None:
      self._state = self._network.initial_state(1)

    # Step the recurrent policy forward given the current observation and state.
    policy_output, new_state = self._policy(observation, self._state)

    # Bookkeeping of recurrent states for the observe method.
    self._prev_state = self._state
    self._state = new_state

    # Return a numpy array with squeezed out batch dimension.
    return tf2_utils.to_numpy_squeeze(policy_output)

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

    # Set the state to None so that we re-initialize at the next policy call.
    self._state = None

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    if not self._adder:
      return

    if not self._store_recurrent_state:
      self._adder.add(action, next_timestep)
      return

    numpy_state = tf2_utils.to_numpy_squeeze(self._prev_state)
    self._adder.add(action, next_timestep, extras=(numpy_state,))

  def update(self, wait: bool = False):
    if self._variable_client:
      self._variable_client.update(wait)

# Internal class 1.
# Internal class 2.
