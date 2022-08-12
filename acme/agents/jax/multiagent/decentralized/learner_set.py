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

"""Decentralized multiagent learnerset."""

import dataclasses
from typing import Any, Dict, List

from acme import core
from acme import types

from acme.multiagent import types as ma_types

LearnerState = Any


@dataclasses.dataclass
class SynchronousDecentralizedLearnerSetState:
  """State of a SynchronousDecentralizedLearnerSet."""
  # States of the learners keyed by their names.
  learner_states: Dict[ma_types.AgentID, LearnerState]


class SynchronousDecentralizedLearnerSet(core.Learner):
  """Creates a composed learner which wraps a set of local agent learners."""

  def __init__(self,
               learners: Dict[ma_types.AgentID, core.Learner],
               separator: str = '-'):
    """Initializer.

    Args:
      learners: a dict specifying the learners for all sub-agents.
      separator: separator character used to disambiguate sub-learner variables.
    """
    self._learners = learners
    self._separator = separator

  def step(self):
    for learner in self._learners.values():
      learner.step()

  def get_variables(self, names: List[str]) -> List[types.NestedArray]:
    """Return the named variables as a collection of (nested) numpy arrays.

    The variable names should be prefixed with the name of the child learners
    using the separator specified in the constructor, e.g. learner1/var.

    Args:
      names: args where each name is a string identifying a predefined subset of
        the variables. The variables names should be prefixed with the name of
        the learners using the separator specified in the constructor, e.g.
        learner-var if the separator is -.

    Returns:
      A list of (nested) numpy arrays `variables` such that `variables[i]`
      corresponds to the collection named by `names[i]`.
    """
    variables = []
    for name in names:
      # Note: if separator is not found, learner_name=name, which is OK.
      learner_id, _, variable_name = name.partition(self._separator)
      learner = self._learners[learner_id]
      variables.extend(learner.get_variables([variable_name]))
    return variables

  def save(self) -> SynchronousDecentralizedLearnerSetState:
    return SynchronousDecentralizedLearnerSetState(learner_states={
        name: learner.save() for name, learner in self._learners.items()
    })

  def restore(self, state: SynchronousDecentralizedLearnerSetState):
    for name, learner in self._learners.items():
      learner.restore(state.learner_states[name])
