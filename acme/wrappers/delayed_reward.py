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

"""Delayed reward wrapper."""

import operator
from typing import Optional

from acme import types
from acme.wrappers import base
import dm_env
import numpy as np
import tree


class DelayedRewardWrapper(base.EnvironmentWrapper):
  """Implements delayed reward on environments.

  This wrapper sparsifies any environment by adding a reward delay. Instead of
  returning a reward at each step, the wrapped environment returns the
  accumulated reward every N steps or at the end of an episode, whichever comes
  first. This does not change the optimal expected return, but typically makes
  the environment harder by adding exploration and longer term dependencies.
  """

  def __init__(self,
               environment: dm_env.Environment,
               accumulation_period: Optional[int] = 1):
    """Initializes a `DelayedRewardWrapper`.

    Args:
      environment: An environment conforming to the dm_env.Environment
        interface.
     accumulation_period: number of steps to accumulate the reward over. If
       `accumulation_period` is an integer, reward is accumulated and returned
       every `accumulation_period` steps, and at the end of an episode. If
       `accumulation_period` is None, reward is only returned at the end of an
       episode. If `accumulation_period`=1, this wrapper is a no-op.
    """

    super().__init__(environment)
    if accumulation_period is not None and accumulation_period < 1:
      raise ValueError(
          f'Accumuluation period is {accumulation_period} but should be greater than 1.'
      )
    self._accumuation_period = accumulation_period
    self._delayed_reward = self._zero_reward
    self._accumulation_counter = 0

  @property
  def _zero_reward(self):
    return tree.map_structure(lambda s: np.zeros(s.shape, s.dtype),
                              self._environment.reward_spec())

  def reset(self) -> dm_env.TimeStep:
    """Resets environment and provides the first timestep."""
    timestep = self.environment.reset()
    self._delayed_reward = self._zero_reward
    self._accumulation_counter = 0
    return timestep

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    """Performs one step and maybe returns a reward."""
    timestep = self.environment.step(action)
    self._delayed_reward = tree.map_structure(operator.iadd,
                                              self._delayed_reward,
                                              timestep.reward)
    self._accumulation_counter += 1

    if (self._accumuation_period is not None and self._accumulation_counter
        == self._accumuation_period) or timestep.last():
      timestep = timestep._replace(reward=self._delayed_reward)
      self._accumulation_counter = 0
      self._delayed_reward = self._zero_reward
    else:
      timestep = timestep._replace(reward=self._zero_reward)

    return timestep
