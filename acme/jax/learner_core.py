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

from typing import Callable, Generic, Mapping

from acme import types
from acme.jax import networks as networks_lib
from acme.jax.types import Sample, TrainingState  # pylint: disable=g-multiple-import
import chex
import dataclasses
import jax.numpy as jnp

Metrics = Mapping[str, jnp.ndarray]

# A mapping of variable collections, as defined by Learner.get_variables.
# The keys are the collection names, the values are nested arrays representing
# the values of the corresponding collection variables.
Variables = Mapping[str, types.NestedArray]


@chex.dataclass(frozen=True, mappable_dataclass=False)
class StepOutput(Generic[TrainingState]):
  state: TrainingState
  metrics: Metrics
  """Metrics returned by the training step.

  Typically these are logged, so the values are expected to be scalars.
  """


# @final
@dataclasses.dataclass
class LearnerCore(Generic[Sample, TrainingState]):
  """Pure functions that define the algorithm-specific learner functionality."""

  init: Callable[[networks_lib.PRNGKey], TrainingState]
  """Initializes the learner state."""

  step: Callable[[TrainingState, Sample], StepOutput[TrainingState]]
  """Does one training step."""

  get_variables: Callable[[TrainingState], Variables]
  """Returns learner variables.

  Extracts the learner variables (both trainable and non-trainable) that need to
  be sent to the actors from the TrainingState.
  """
