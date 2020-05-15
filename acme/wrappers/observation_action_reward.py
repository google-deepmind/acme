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

"""A wrapper that puts the previous action and reward into the observation."""

from typing import NamedTuple

from acme import types
from acme.wrappers import base

import dm_env
import tree


class OAR(NamedTuple):
  """Container for (Observation, Action, Reward) tuples."""
  observation: types.Nest
  action: types.Nest
  reward: types.Nest


class ObservationActionRewardWrapper(base.EnvironmentWrapper):
  """A wrapper that puts the previous action and reward into the observation."""

  _prev_action: types.NestedArray
  _prev_reward: types.NestedArray

  def reset(self) -> dm_env.TimeStep:
    # Initialize with zeros of the appropriate shape/dtype.
    self._prev_action = tree.map_structure(
        lambda x: x.generate_value(), self._environment.action_spec())
    self._prev_reward = tree.map_structure(
        lambda x: x.generate_value(), self._environment.reward_spec())
    timestep = self._environment.reset()
    new_timestep = self._augment_observation(timestep)
    return new_timestep

  def step(self, action) -> dm_env.TimeStep:
    timestep = self._environment.step(action)
    new_timestep = self._augment_observation(timestep)
    self._prev_action = action
    self._prev_reward = timestep.reward
    return new_timestep

  def _augment_observation(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    oar = OAR(observation=timestep.observation,
              action=self._prev_action,
              reward=self._prev_reward)
    return timestep._replace(observation=oar)

  def observation_spec(self):
    return OAR(observation=self._environment.observation_spec(),
               action=self.action_spec(),
               reward=self.reward_spec())
