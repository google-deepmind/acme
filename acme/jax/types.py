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

"""Common JAX type definitions."""

from typing import Callable, Generic, Mapping, TypeVar

from acme import core
from acme import types
from acme.utils import counting
import chex
import jax.numpy as jnp

PRNGKey = jnp.ndarray
EvaluatorFactory = Callable[[PRNGKey, core.VariableSource, counting.Counter],
                            core.Worker]
Networks = TypeVar('Networks')
PolicyNetwork = TypeVar('PolicyNetwork')
Sample = TypeVar('Sample')
TrainingState = TypeVar('TrainingState')

TrainingMetrics = Mapping[str, jnp.ndarray]
"""Metrics returned by the training step.

Typically these are logged, so the values are expected to be scalars.
"""

Variables = Mapping[str, types.NestedArray]
"""Mapping of variable collections.

A mapping of variable collections, as defined by Learner.get_variables.
The keys are the collection names, the values are nested arrays representing
the values of the corresponding collection variables.
"""


@chex.dataclass(frozen=True, mappable_dataclass=False)
class TrainingStepOutput(Generic[TrainingState]):
  state: TrainingState
  metrics: TrainingMetrics
