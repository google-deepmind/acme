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

"""Frozen learner."""

from typing import Callable, List, Optional, Sequence

import acme


class FrozenLearner(acme.Learner):
  """Wraps a learner ignoring the step calls, i.e. freezing it."""

  def __init__(self,
               learner: acme.Learner,
               step_fn: Optional[Callable[[], None]] = None):
    """Initializes the frozen learner.

    Args:
      learner: Learner to be wrapped.
      step_fn: Function to call instead of the step() method of the learner.
        This can be used, e.g. to drop samples from an iterator that would
        normally be consumed by the learner.
    """
    self._learner = learner
    self._step_fn = step_fn

  def step(self):
    """See base class."""
    if self._step_fn:
      self._step_fn()

  def run(self, num_steps: Optional[int] = None):
    """See base class."""
    self._learner.run(num_steps)

  def save(self):
    """See base class."""
    return self._learner.save()

  def restore(self, state):
    """See base class."""
    self._learner.restore(state)

  def get_variables(self, names: Sequence[str]) -> List[acme.types.NestedArray]:
    """See base class."""
    return self._learner.get_variables(names)
