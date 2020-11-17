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
  def _policy(self, observation: types.NestedTensor,
              legal_actions: types.NestedTensor) -> types.NestedTensor:
    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_observation = tf2_utils.add_batch_dim(observation)

    # Compute the policy, conditioned on the observation.
    policy = self._policy_network(batched_observation, legal_actions)

    # Sample from the policy if it is stochastic.
    action = policy.sample() if isinstance(policy, tfd.Distribution) else policy

    return action

  def select_action(
      self,
      observation: types.NestedArray,
      legal_actions: Optional[types.NestedTensor] = None) -> types.NestedArray:
    # Pass the observation through the policy network.
    action = self._policy(observation, legal_actions)

    # Return a numpy array with squeezed out batch dimension.
    return tf2_utils.to_numpy_squeeze(action)

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

  def observe(self,
              action: types.NestedArray,
              next_timestep: dm_env.TimeStep,
              extras: types.NestedArray = ()):
    if self._adder:
      self._adder.add(action, next_timestep, extras)

  def update(self, wait: bool = False):
    if self._variable_client:
      self._variable_client.update(wait)
