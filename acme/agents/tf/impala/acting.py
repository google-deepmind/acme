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

"""IMPALA actor implementation."""

from acme import adders
from acme import core
from acme import types
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

import dm_env
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class IMPALAActor(core.Actor):
  """A recurrent actor."""

  def __init__(
      self,
      network: snt.RNNCore,
      adder: adders.Adder = None,
      variable_client: tf2_variable_utils.VariableClient = None,
  ):

    # Store these for later use.
    self._adder = adder
    self._variable_client = variable_client
    self._network = network

    # TODO(b/152382420): Ideally we would call tf.function(network) instead but
    # this results in an error when using acme RNN snapshots.
    self._policy = tf.function(network.__call__)

    self._state = None
    self._prev_state = None
    self._prev_logits = None

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_obs = tf2_utils.add_batch_dim(observation)

    if self._state is None:
      self._state = self._network.initial_state(1)

    # Forward.
    (logits, _), new_state = self._policy(batched_obs, self._state)

    self._prev_logits = logits
    self._prev_state = self._state
    self._state = new_state

    action = tfd.Categorical(logits).sample()
    action = tf2_utils.to_numpy_squeeze(action)

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

    extras = {'logits': self._prev_logits, 'core_state': self._prev_state}
    extras = tf2_utils.to_numpy_squeeze(extras)
    self._adder.add(action, next_timestep, extras)

  def update(self):
    if self._variable_client:
      self._variable_client.update()
