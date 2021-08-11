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

"""SVG0 actor implementation."""

from typing import Optional

from acme import adders
from acme import types

from acme.agents.tf import actors
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

import dm_env
import sonnet as snt


class SVG0Actor(actors.FeedForwardActor):
  """An actor that also returns `log_prob`."""

  def __init__(
      self,
      policy_network: snt.Module,
      adder: Optional[adders.Adder] = None,
      variable_client: Optional[tf2_variable_utils.VariableClient] = None,
      deterministic_policy: Optional[bool] = False,
  ):
    super().__init__(policy_network, adder, variable_client)
    self._log_prob = None
    self._deterministic_policy = deterministic_policy

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_observation = tf2_utils.add_batch_dim(observation)

    # Compute the policy, conditioned on the observation.
    policy = self._policy_network(batched_observation)
    if self._deterministic_policy:
      action = policy.mean()
    else:
      action = policy.sample()
    self._log_prob = policy.log_prob(action)
    return tf2_utils.to_numpy_squeeze(action)

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    if not self._adder:
      return

    extras = {'log_prob': self._log_prob}
    extras = tf2_utils.to_numpy_squeeze(extras)
    self._adder.add(action, next_timestep, extras)
