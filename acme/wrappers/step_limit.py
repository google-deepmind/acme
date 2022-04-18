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

"""Wrapper that implements environment step limit."""

from typing import Optional
from acme import types
from acme.wrappers import base
import dm_env


class StepLimitWrapper(base.EnvironmentWrapper):
  """A wrapper which truncates episodes at the specified step limit."""

  def __init__(self, environment: dm_env.Environment,
               step_limit: Optional[int] = None):
    super().__init__(environment)
    self._step_limit = step_limit
    self._elapsed_steps = 0

  def reset(self) -> dm_env.TimeStep:
    self._elapsed_steps = 0
    return self._environment.reset()

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    timestep = self._environment.step(action)
    self._elapsed_steps += 1
    if self._step_limit is not None and self._elapsed_steps >= self._step_limit:
      return dm_env.truncation(
          timestep.reward, timestep.observation, timestep.discount)
    return timestep
