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

"""Wrapper that implements concatenation of observation fields."""

from typing import Sequence, Optional

from acme import types
from acme.wrappers import base
import dm_env
import numpy as np
import tree


def _concat(values: types.NestedArray) -> np.ndarray:
  """Concatenates the leaves of `values` along the leading dimension.

  Treats scalars as 1d arrays and expects that the shapes of all leaves are
  the same except for the leading dimension.

  Args:
    values: the nested arrays to concatenate.

  Returns:
    The concatenated array.
  """
  leaves = list(map(np.atleast_1d, tree.flatten(values)))
  return np.concatenate(leaves)


def _zeros_like(nest, dtype=None):
  """Generate a nested NumPy array according to spec."""
  return tree.map_structure(lambda x: np.zeros(x.shape, dtype or x.dtype), nest)


class ConcatObservationWrapper(base.EnvironmentWrapper):
  """Wrapper that concatenates observation fields.

  It takes an environment with nested observations and concatenates the fields
  in a single tensor. The original fields should be 1-dimensional.
  Observation fields that are not in name_filter are dropped.

  **NOTE**: The fields in the flattened observations will be in sorted order by
  their names, see tree.flatten for more information.
  """

  def __init__(self,
               environment: dm_env.Environment,
               name_filter: Optional[Sequence[str]] = None):
    """Initializes a new ConcatObservationWrapper.

    Args:
      environment: Environment to wrap.
      name_filter: Sequence of observation names to keep. None keeps them all.
    """
    super().__init__(environment)
    observation_spec = environment.observation_spec()
    if name_filter is None:
      name_filter = list(observation_spec.keys())
    self._obs_names = [x for x in name_filter if x in observation_spec.keys()]

    dummy_obs = _zeros_like(observation_spec)
    dummy_obs = self._convert_observation(dummy_obs)
    self._observation_spec = dm_env.specs.BoundedArray(
        shape=dummy_obs.shape,
        dtype=dummy_obs.dtype,
        minimum=-np.inf,
        maximum=np.inf,
        name='state')

  def _convert_observation(self, observation):
    obs = {k: observation[k] for k in self._obs_names}
    return _concat(obs)

  def step(self, action) -> dm_env.TimeStep:
    timestep = self._environment.step(action)
    return timestep._replace(
        observation=self._convert_observation(timestep.observation))

  def reset(self) -> dm_env.TimeStep:
    timestep = self._environment.reset()
    return timestep._replace(
        observation=self._convert_observation(timestep.observation))

  def observation_spec(self) -> types.NestedSpec:
    return self._observation_spec
