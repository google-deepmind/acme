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

"""Frame stacking utilities."""

import collections

from acme import types
from acme.wrappers import base
import dm_env
from dm_env import specs as dm_env_specs
import numpy as np
import tree


class FrameStackingWrapper(base.EnvironmentWrapper):
  """Wrapper that stacks observations along a new final axis."""

  def __init__(self, environment: dm_env.Environment, num_frames: int = 4,
               flatten: bool = False):
    """Initializes a new FrameStackingWrapper.

    Args:
      environment: Environment.
      num_frames: Number frames to stack.
      flatten: Whether to flatten the channel and stack dimensions together.
    """
    self._environment = environment
    original_spec = self._environment.observation_spec()
    self._stackers = tree.map_structure(
        lambda _: FrameStacker(num_frames=num_frames, flatten=flatten),
        self._environment.observation_spec())
    self._observation_spec = tree.map_structure(
        lambda stacker, spec: stacker.update_spec(spec),
        self._stackers, original_spec)

  def _process_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    observation = tree.map_structure(lambda stacker, x: stacker.step(x),
                                     self._stackers, timestep.observation)
    return timestep._replace(observation=observation)

  def reset(self) -> dm_env.TimeStep:
    for stacker in tree.flatten(self._stackers):
      stacker.reset()
    return self._process_timestep(self._environment.reset())

  def step(self, action: int) -> dm_env.TimeStep:
    return self._process_timestep(self._environment.step(action))

  def observation_spec(self) -> types.NestedSpec:
    return self._observation_spec


class FrameStacker:
  """Simple class for frame-stacking observations."""

  def __init__(self, num_frames: int, flatten: bool = False):
    self._num_frames = num_frames
    self._flatten = flatten
    self.reset()

  @property
  def num_frames(self) -> int:
    return self._num_frames

  def reset(self):
    self._stack = collections.deque(maxlen=self._num_frames)

  def step(self, frame: np.ndarray) -> np.ndarray:
    """Append frame to stack and return the stack."""
    if not self._stack:
      # Fill stack with blank frames if empty.
      self._stack.extend([np.zeros_like(frame)] * (self._num_frames - 1))
    self._stack.append(frame)
    stacked_frames = np.stack(self._stack, axis=-1)

    if not self._flatten:
      return stacked_frames
    else:
      new_shape = stacked_frames.shape[:-2] + (-1,)
      return stacked_frames.reshape(*new_shape)

  def update_spec(self, spec: dm_env_specs.Array) -> dm_env_specs.Array:
    if not self._flatten:
      new_shape = spec.shape + (self._num_frames,)
    else:
      new_shape = spec.shape[:-1] + (self._num_frames * spec.shape[-1],)
    return dm_env_specs.Array(shape=new_shape, dtype=spec.dtype, name=spec.name)
