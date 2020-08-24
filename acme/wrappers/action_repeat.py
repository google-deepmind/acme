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

"""Wrapper that implements action repeats."""

from acme import types
from acme.wrappers import base
import dm_env


class ActionRepeatWrapper(base.EnvironmentWrapper):
  """Action repeat wrapper."""

  def __init__(self, environment: dm_env.Environment, num_repeats: int = 1):
    super().__init__(environment)
    self._num_repeats = num_repeats

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    # Initialize accumulated reward and discount.
    reward = 0.
    discount = 1.

    # Step the environment by repeating action.
    for _ in range(self._num_repeats):
      timestep = self._environment.step(action)

      # Accumulate reward and discount.
      reward += timestep.reward * discount
      discount *= timestep.discount

      # Don't go over episode boundaries.
      if timestep.last():
        break

    # Replace the final timestep's reward and discount.
    return timestep._replace(reward=reward, discount=discount)
