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

"""The base agent interface."""

from typing import List

from acme import core
from acme import types
# Internal imports.

import dm_env
import numpy as np


class Agent(core.Actor, core.VariableSource):
  """Agent class which combines acting and learning.

  This provides an implementation of the `Actor` interface which acts and
  learns. It takes as input instances of both `acme.Actor` and `acme.Learner`
  classes, and implements the policy, observation, and update methods which
  defer to the underlying actor and learner.

  The only real logic implemented by this class is that it controls the number
  of observations to make before running a learner step. This is done by
  passing the number of `min_observations` to use and a ratio of
  `observations_per_step`

  Note that the number of `observations_per_step` which can also be in the range
  [0, 1] in order to allow more steps per update.
  """

  def __init__(self, actor: core.Actor, learner: core.Learner,
               min_observations: int, observations_per_step: float):
    self._actor = actor
    self._learner = learner

    # We'll ignore the first min_observations when determining whether to take
    # a step and we'll do so by making sure num_observations >= 0.
    self._num_observations = -min_observations

    # Rather than work directly with the observations_per_step ratio we can
    # figure out how many observations or steps to run per update, one of which
    # should be one.
    if observations_per_step >= 1.0:
      self._observations_per_update = int(observations_per_step)
      self._steps_per_update = 1
    else:
      self._observations_per_update = 1
      self._steps_per_update = int(1.0 / observations_per_step)

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    return self._actor.select_action(observation)

  def observe_first(self, timestep: dm_env.TimeStep):
    self._actor.observe_first(timestep)

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    self._num_observations += 1
    self._actor.observe(action, next_timestep)

  def update(self):
    # Only allow updates after some minimum number of observations have been and
    # then at some period given by observations_per_update.
    if (self._num_observations >= 0 and
        self._num_observations % self._observations_per_update == 0):
      self._num_observations = 0

      # Run a number of learner steps (usually gradient steps).
      for _ in range(self._steps_per_update):
        self._learner.step()
      # Update actor weights after learner, note in TF this may be a no-op.
      self._actor.update()

  def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
    return self._learner.get_variables(names)


# Internal class.
