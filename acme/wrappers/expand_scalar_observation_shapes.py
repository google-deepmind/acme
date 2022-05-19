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

"""This wrapper expands scalar observations to have non-trivial shape.

This is useful for example if the observation holds the previous (scalar)
action, which can cause issues when manipulating arrays with axis=-1. This
wrapper makes sure the environment returns a previous action with shape [1].

This can be necessary when stacking observations with previous actions.
"""

from typing import Any

from acme.wrappers import base
import dm_env
from dm_env import specs
import numpy as np
import tree


class ExpandScalarObservationShapesWrapper(base.EnvironmentWrapper):
  """Expands scalar shapes in the observation.

  For example, if the observation holds the previous (scalar) action, this
  wrapper makes sure the environment returns a previous action with shape [1].

  This can be necessary when stacking observations with previous actions.
  """

  def step(self, action: Any) -> dm_env.TimeStep:
    timestep = self._environment.step(action)
    expanded_observation = tree.map_structure(_expand_scalar_array_shape,
                                              timestep.observation)
    return timestep._replace(observation=expanded_observation)

  def reset(self) -> dm_env.TimeStep:
    timestep = self._environment.reset()
    expanded_observation = tree.map_structure(_expand_scalar_array_shape,
                                              timestep.observation)
    return timestep._replace(observation=expanded_observation)

  def observation_spec(self) -> specs.Array:
    return tree.map_structure(_expand_scalar_spec_shape,
                              self._environment.observation_spec())


def _expand_scalar_spec_shape(spec: specs.Array) -> specs.Array:
  if not spec.shape:
    # NOTE: This line upcasts the spec to an Array to avoid edge cases (as in
    # DiscreteSpec) where we cannot set the spec's shape.
    spec = specs.Array(shape=(1,), dtype=spec.dtype, name=spec.name)
  return spec


def _expand_scalar_array_shape(array: np.ndarray) -> np.ndarray:
  return array if array.shape else np.expand_dims(array, axis=-1)
