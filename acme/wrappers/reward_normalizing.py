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

"""A wrapper that scales the rewards wrt an observed score."""

from acme import types
from acme.wrappers import base

import dm_env


class RewardNormalizingWrapper(base.EnvironmentWrapper):
  """A wrapper that normalizes the rewards wrt observed min and max scores."""

  def __init__(
      self, environment: dm_env.Environment, r_min: float, r_max: float):
    super(RewardNormalizingWrapper, self).__init__(environment)
    self._shift = r_min
    if r_min >= r_max:
      raise ValueError(f'r_min={r_min} >= r_max={r_max}')
    self._scale = 1 / (r_max - r_min)

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    timestep = self._environment.step(action)
    normalized_reward = (timestep.reward - self._shift) * self._scale
    return timestep._replace(reward=normalized_reward)
