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

import math
from typing import List, Optional

from acme import core
from acme import types
import dm_env
import numpy as np
import reverb


def _calculate_num_learner_steps(num_observations: int,
                                 min_observations: int,
                                 observations_per_step: float) -> int:
  """Calculates the number of learner steps to do at step=num_observations."""
  n = num_observations - min_observations
  if n < 0:
    # Do not do any learner steps until you have seen min_observations.
    return 0
  if observations_per_step > 1:
    # One batch every 1/obs_per_step observations, otherwise zero.
    return int(n % int(observations_per_step) == 0)
  else:
    # Always return 1/obs_per_step batches every observation.
    return int(1 / observations_per_step)


class Agent(core.Actor, core.VariableSource):
  """Agent class which combines acting and learning.

  This provides an implementation of the `Actor` interface which acts and
  learns. It takes as input instances of both `acme.Actor` and `acme.Learner`
  classes, and implements the policy, observation, and update methods which
  defer to the underlying actor and learner.

  The only real logic implemented by this class is that it controls the number
  of observations to make before running a learner step. This is done by
  passing the number of `min_observations` to use and a ratio of
  `observations_per_step` := num_actor_actions / num_learner_steps.

  Note that the number of `observations_per_step` can also be in the range[0, 1]
  in order to allow the agent to take more than 1 learner step per action.
  """

  def __init__(self, actor: core.Actor, learner: core.Learner,
               min_observations: Optional[int] = None,
               observations_per_step: Optional[float] = None,
               iterator: Optional[core.PrefetchingIterator] = None,
               replay_tables: Optional[List[reverb.Table]] = None):
    self._actor = actor
    self._learner = learner
    self._min_observations = min_observations
    self._observations_per_step = observations_per_step
    self._num_observations = 0
    self._iterator = iterator
    self._replay_tables = replay_tables
    self._batch_size_upper_bounds = [1_000_000_000] * len(
        replay_tables) if replay_tables else None

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    return self._actor.select_action(observation)

  def observe_first(self, timestep: dm_env.TimeStep):
    self._actor.observe_first(timestep)

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    self._num_observations += 1
    self._actor.observe(action, next_timestep)

  def _has_data_for_training(self):
    if self._iterator.ready():
      return True
    for (table, batch_size) in zip(self._replay_tables,
                                   self._batch_size_upper_bounds):
      if not table.can_sample(batch_size):
        return False
    return True

  def update(self):
    if self._iterator:
      # Perform learner steps as long as iterator has data.
      update_actor = False
      while self._has_data_for_training():
        # Run learner steps (usually means gradient steps).
        total_batches = self._iterator.retrieved_elements()
        self._learner.step()
        current_batches = self._iterator.retrieved_elements() - total_batches
        assert current_batches == 1, (
            'Learner step must retrieve exactly one element from the iterator'
            f' (retrieved {current_batches}). Otherwise agent can deadlock. '
            'Example cause is that your chosen agent'
            's Builder has a '
            '`make_learner` factory that prefetches the data but it '
            'shouldn'
            't.')
        self._batch_size_upper_bounds = [
            math.ceil(t.info.rate_limiter_info.sample_stats.completed /
                      (total_batches + 1)) for t in self._replay_tables
        ]
        update_actor = True
      if update_actor:
        # Update the actor weights only when learner was updated.
        self._actor.update()
      return

    # If dataset is not provided, follback to the old logic.
    # TODO(stanczyk): Remove when not used.
    num_steps = _calculate_num_learner_steps(
        num_observations=self._num_observations,
        min_observations=self._min_observations,
        observations_per_step=self._observations_per_step,
    )
    for _ in range(num_steps):
      # Run learner steps (usually means gradient steps).
      self._learner.step()
    if num_steps > 0:
      # Update the actor weights when learner updates.
      self._actor.update()

  def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
    return self._learner.get_variables(names)
