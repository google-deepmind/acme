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
from typing import List, Optional, Sequence

from acme import core
from acme import types
import dm_env
import numpy as np
import reverb

_MAX_BATCH_SIZE_UPPER_BOUND = 1_000_000_000

def _calculate_num_learner_steps(
    num_observations: int,
    min_observations: int,
    observations_per_step: float
) -> int:
    """Calculate the number of learner steps to run at a given number of observations.

    Args:
        num_observations: The total number of observations collected so far.
        min_observations: The minimum number of observations required before learning starts.
        observations_per_step: Ratio of observations per learner step. Can be >1 or <1.

    Returns:
        The number of learner steps to run at this observation count.
    """
    steps_since_min = num_observations - min_observations
    if steps_since_min < 0:
        # Not enough observations collected yet.
        return 0

    if observations_per_step > 1:
        # Run one learner step every 'observations_per_step' observations.
        return int(steps_since_min % int(observations_per_step) == 0)
    else:
        # Run multiple learner steps per observation.
        return int(1 / observations_per_step)


class Agent(core.Actor, core.VariableSource):
    """Agent that combines acting and learning by composing an Actor and Learner.

    This class controls the frequency of learner updates relative to actor steps,
    either using an iterator or a simple ratio-based approach.

    Args:
        actor: An instance implementing the Actor interface.
        learner: An instance implementing the Learner interface.
        min_observations: Minimum number of observations before learning starts.
        observations_per_step: Number of observations per learner step ratio.
        iterator: Optional data iterator for learner.
        replay_tables: Optional replay tables used for sampling data.
    """

    def __init__(
        self,
        actor: core.Actor,
        learner: core.Learner,
        min_observations: Optional[int] = None,
        observations_per_step: Optional[float] = None,
        iterator: Optional[core.PrefetchingIterator] = None,
        replay_tables: Optional[List[reverb.Table]] = None,
    ):
        self._actor = actor
        self._learner = learner
        self._min_observations = min_observations
        self._observations_per_step = observations_per_step 
        self._num_observations = 0
        self._iterator = iterator
        self._replay_tables = replay_tables or []
        self._batch_size_upper_bounds = [
            _MAX_BATCH_SIZE_UPPER_BOUND for _ in self._replay_tables
        ] if self._replay_tables else None

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        return self._actor.select_action(observation)

    def observe_first(self, timestep: dm_env.TimeStep):
        self._actor.observe_first(timestep)

    def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
        self._num_observations += 1
        self._actor.observe(action, next_timestep)

    def _has_data_for_training(self, iterator: core.PrefetchingIterator) -> bool:
        """Check if there is sufficient data to run learner steps."""
        if iterator.ready():
            return True

        for table, batch_size in zip(self._replay_tables, self._batch_size_upper_bounds):
            if not table.can_sample(batch_size):
                return False
        return True

    def update(self):
        """Perform learner updates if enough data is available and update the actor weights."""
        if self._iterator:
            update_actor = False
            while self._has_data_for_training(self._iterator):
                total_batches = self._iterator.retrieved_elements()
                self._learner.step()
                current_batches = self._iterator.retrieved_elements() - total_batches
                assert current_batches == 1, (
                    f'Learner step must retrieve exactly one element from the iterator '
                    f'(retrieved {current_batches}). Otherwise agent can deadlock. '
                    'Example cause: your agent\'s Builder uses a `make_learner` factory '
                    'that prefetches data but it shouldn\'t.'
                )
                self._batch_size_upper_bounds = [
                    math.ceil(
                        t.info.rate_limiter_info.sample_stats.completed / (total_batches + 1)
                    ) for t in self._replay_tables
                ]
                update_actor = True

            if update_actor:
                self._actor.update()
            return

        # Fallback if no iterator is provided.
        num_steps = _calculate_num_learner_steps(
            num_observations=self._num_observations,
            min_observations=self._min_observations,
            observations_per_step=self._observations_per_step,
        )
        for _ in range(num_steps):
            self._learner.step()
        if num_steps > 0:
            self._actor.update()

    def get_variables(self, names: Sequence[str]) -> List[List[np.ndarray]]:
        return self._actor.get_variables(names)
