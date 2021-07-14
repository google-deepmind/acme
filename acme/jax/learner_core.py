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

"""LearnerCore interface definition."""

from typing import Callable, Generic

from acme.jax.types import PRNGKey, Sample, TrainingState, TrainingStepOutput, Variables  # pylint: disable=g-multiple-import
import dataclasses


# @final
@dataclasses.dataclass
class LearnerCore(Generic[Sample, TrainingState]):
  """Pure functions that define the algorithm-specific learner functionality."""

  init: Callable[[PRNGKey], TrainingState]
  """Initializes the learner state."""

  step: Callable[[TrainingState, Sample], TrainingStepOutput[TrainingState]]
  """Does one training step."""

  get_variables: Callable[[TrainingState], Variables]
  """Returns learner variables.

  Extracts the learner variables (both trainable and non-trainable) that need to
  be sent to the actors from the TrainingState.
  """
