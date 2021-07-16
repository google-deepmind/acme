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

"""Generic Acme learner definitions for JAX."""

import time
from typing import Generic, Iterator, List, Optional

import acme
from acme import types
from acme.jax import learner_core as learner_core_lib
from acme.jax import networks as networks_lib
from acme.jax.types import Sample, TrainingState  # pylint: disable=g-multiple-import
from acme.utils import counting
from acme.utils import loggers
import jax


# @final
class DefaultJaxLearner(acme.Learner, Generic[Sample, TrainingState]):
  """Default JAX learner."""

  def __init__(
      self,
      learner_core: learner_core_lib.LearnerCore[Sample, TrainingState],
      iterator: Iterator[Sample],
      random_key: networks_lib.PRNGKey,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
  ):

    # Initialise training state (parameters and optimiser state).
    self._state = learner_core.init(random_key)

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

    # Internalise iterator.
    self._iterator = iterator
    self._sgd_step = jax.jit(learner_core.step)
    self._get_variables = learner_core.get_variables

    # Set up logging/counting.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger('learner')

  def step(self):
    """Does a step of SGD and logs the results."""

    # Do a batch of SGD.
    sample = next(self._iterator)
    step_output = self._sgd_step(self._state, sample)
    self._state, metrics = step_output.state, step_output.metrics

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Update our counts and record it.
    counts = self._counter.increment(steps=1, walltime=elapsed_time)

    # Snapshot and attempt to write logs.
    self._logger.write({**metrics, **counts})

  def get_variables(self, names: List[str]) -> List[types.NestedArray]:
    variables = self._get_variables(self._state)
    return [variables.get(name, None) for name in names]

  def save(self) -> TrainingState:
    return self._state

  def restore(self, state: TrainingState):
    self._state = state
