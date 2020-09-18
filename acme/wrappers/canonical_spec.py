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

"""Environment wrapper that converts environments to use canonical action specs.

This only affects action specs of type `specs.BoundedArray`.

For bounded action specs, we refer to a canonical action spec as the bounding
box [-1, 1]^d where d is the dimensionality of the spec. So the shape and dtype
of the spec is unchanged, while the maximum/minimum values are set to +/- 1.
"""

from acme import specs
from acme import types
from acme.wrappers import base

import dm_env
import numpy as np
import tree


class CanonicalSpecWrapper(base.EnvironmentWrapper):
  """Wrapper which converts environments to use canonical action specs.

  This only affects action specs of type `specs.BoundedArray`.

  For bounded action specs, we refer to a canonical action spec as the bounding
  box [-1, 1]^d where d is the dimensionality of the spec. So the shape and
  dtype of the spec is unchanged, while the maximum/minimum values are set
  to +/- 1.
  """

  def __init__(self, environment: dm_env.Environment, clip: bool = False):
    super().__init__(environment)
    self._action_spec = environment.action_spec()
    self._clip = clip

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    scaled_action = _scale_nested_action(action, self._action_spec, self._clip)
    return self._environment.step(scaled_action)

  def action_spec(self):
    return _convert_spec(self._environment.action_spec())


def _convert_spec(nested_spec: types.NestedSpec) -> types.NestedSpec:
  """Converts all bounded specs in nested spec to the canonical scale."""

  def _convert_single_spec(spec: specs.Array) -> specs.Array:
    """Converts a single spec to canonical if bounded."""
    if isinstance(spec, specs.BoundedArray):
      return spec.replace(
          minimum=-np.ones(spec.shape), maximum=np.ones(spec.shape))
    else:
      return spec

  return tree.map_structure(_convert_single_spec, nested_spec)


def _scale_nested_action(
    nested_action: types.NestedArray,
    nested_spec: types.NestedSpec,
    clip: bool,
) -> types.NestedArray:
  """Converts a canonical nested action back to the given nested action spec."""

  def _scale_action(action: np.ndarray, spec: specs.Array):
    """Converts a single canonical action back to the given action spec."""
    if isinstance(spec, specs.BoundedArray):
      # Get scale and offset of output action spec.
      scale = spec.maximum - spec.minimum
      offset = spec.minimum

      # Maybe clip the action.
      if clip:
        action = np.clip(action, -1.0, 1.0)

      # Map action to [0, 1].
      action = 0.5 * (action + 1.0)

      # Map action to [spec.minimum, spec.maximum].
      action *= scale
      action += offset

    return action

  return tree.map_structure(_scale_action, nested_action, nested_spec)
