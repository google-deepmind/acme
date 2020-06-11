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

"""Generic actor implementation, using TensorFlow and Sonnet."""

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
import tree

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
      adder: adders.Adder = None,
      variable_client: tf2_variable_utils.VariableClient = None,
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
    self._policy_network = tf.function(policy_network)

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_obs = tf2_utils.add_batch_dim(observation)

    # Forward the policy network.
    policy_output = self._policy_network(batched_obs)

    # If the policy network parameterises a distribution, sample from it.
    def maybe_sample(output):
      if isinstance(output, tfd.Distribution):
        output = output.sample()
      return output

    policy_output = tree.map_structure(maybe_sample, policy_output)

    # Convert to numpy and squeeze out the batch dimension.
    action = tf2_utils.to_numpy_squeeze(policy_output)

    return action

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    if self._adder:
      self._adder.add(action, next_timestep)

  def update(self):
    if self._variable_client:
      self._variable_client.update()


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
      adder: adders.Adder = None,
      variable_client: tf2_variable_utils.VariableClient = None,
  ):
    """Initializes the actor.

    Args:
      policy_network: the (recurrent) policy to run.
      adder: the adder object to which allows to add experiences to a
        dataset/replay buffer.
      variable_client: object which allows to copy weights from the learner copy
        of the policy to the actor copy (in case they are separate).
    """
    # Store these for later use.
    self._adder = adder
    self._variable_client = variable_client
    self._network = policy_network
    self._state = None
    self._prev_state = None

    # TODO(b/152382420): Ideally we would call tf.function(network) instead but
    # this results in an error when using acme RNN snapshots.
    self._policy = tf.function(policy_network.__call__)

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_obs = tf2_utils.add_batch_dim(observation)

    # Initialize the RNN state if necessary.
    if self._state is None:
      self._state = self._network.initial_state(1)

    # Forward.
    policy_output, new_state = self._policy(batched_obs, self._state)

    # If the policy network parameterises a distribution, sample from it.
    def maybe_sample(output):
      if isinstance(output, tfd.Distribution):
        output = output.sample()
      return output

    policy_output = tree.map_structure(maybe_sample, policy_output)

    self._prev_state = self._state
    self._state = new_state

    # Convert to numpy and squeeze out the batch dimension.
    action = tf2_utils.to_numpy_squeeze(policy_output)

    return action

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

    # Set the state to None so that we re-initialize at the next policy call.
    self._state = None

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    if not self._adder:
      return

    numpy_state = tf2_utils.to_numpy_squeeze(self._prev_state)
    self._adder.add(action, next_timestep, extras=(numpy_state,))

  def update(self):
    if self._variable_client:
      self._variable_client.update()


# Internal class 1.
# Internal class 2.
