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

"""Reward-substituting adder wrapper."""

from acme import adders
from acme import types
from acme.agents.jax.pwil import rewarder
import dm_env


class PWILAdder(adders.Adder):
  """Adder wrapper substituting PWIL rewards."""

  def __init__(self, direct_rl_adder: adders.Adder,
               pwil_rewarder: rewarder.WassersteinDistanceRewarder):
    self._adder = direct_rl_adder
    self._rewarder = pwil_rewarder
    self._latest_observation = None

  def add_first(self, timestep: dm_env.TimeStep):
    self._rewarder.reset()
    self._latest_observation = timestep.observation
    self._adder.add_first(timestep)

  def add(self,
          action: types.NestedArray,
          next_timestep: dm_env.TimeStep,
          extras: types.NestedArray = ()):
    updated_timestep = next_timestep._replace(
        reward=self._rewarder.append_and_compute_reward(
            observation=self._latest_observation, action=action))
    self._latest_observation = next_timestep.observation
    self._adder.add(action, updated_timestep, extras)

  def reset(self):
    self._latest_observation = None
    self._adder.reset()
