# Copyright 2024 DeepMind Technologies Limited. All rights reserved.
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

"""Wraps a Farama Gymnasium environment to be used as a dm_env environment."""

from typing import Any, Dict, Optional

from acme import specs
from acme import types

import dm_env
import numpy as np
import tree

try:
  import gymnasium
  from gymnasium import spaces
  _HAS_GYMNASIUM = True
except (ImportError, ModuleNotFoundError):
  _HAS_GYMNASIUM = False
  gymnasium = None
  spaces = None


class GymnasiumWrapper(dm_env.Environment):
  """Environment wrapper for Farama Gymnasium environments."""

  def __init__(self, environment: Any):
    if not _HAS_GYMNASIUM:
      raise ImportError(
          'Gymnasium is not installed. Please install it with `pip install gymnasium`.'
      )
    self._environment = environment
    self._reset_next_step = True
    self._last_info: Optional[Dict[str, Any]] = None

    obs_space = self._environment.observation_space
    act_space = self._environment.action_space
    self._observation_spec = _convert_to_spec(obs_space, name='observation')
    self._action_spec = _convert_to_spec(act_space, name='action')

  def reset(
      self,
      seed: Optional[int] = None,
      options: Optional[Dict[str, Any]] = None,
  ) -> dm_env.TimeStep:
    """Resets the episode."""
    self._reset_next_step = False
    kwargs = {}
    if seed is not None:
      kwargs['seed'] = seed
    if options is not None:
      kwargs['options'] = options
    observation, info = self._environment.reset(**kwargs)
    self._last_info = info
    return dm_env.restart(observation)

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    """Steps the environment."""
    if self._reset_next_step:
      return self.reset()

    observation, reward, terminated, truncated, info = self._environment.step(action)
    self._reset_next_step = terminated or truncated
    self._last_info = info

    reward = tree.map_structure(
        lambda x, t: (
            t.dtype.type(x)
            if np.isscalar(x) else np.asarray(x, dtype=t.dtype)),
        reward,
        self.reward_spec())

    if truncated:
      return dm_env.truncation(reward, observation)
    if terminated:
      return dm_env.termination(reward, observation)
    return dm_env.transition(reward, observation)

  def observation_spec(self) -> types.NestedSpec:
    return self._observation_spec

  def action_spec(self) -> types.NestedSpec:
    return self._action_spec

  def get_info(self) -> Optional[Dict[str, Any]]:
    """Returns the last info returned from env.step(action) or env.reset()."""
    return self._last_info

  @property
  def environment(self) -> Any:
    """Returns the wrapped environment."""
    return self._environment

  def __getattr__(self, name: str):
    if name.startswith('__'):
      raise AttributeError(
          f"attempted to get missing private attribute '{name}'")
    return getattr(self._environment, name)

  def close(self):
    self._environment.close()


def _convert_to_spec(space: Any,
                     name: Optional[str] = None) -> types.NestedSpec:
  """Converts a Gymnasium space to a dm_env spec or nested structure of specs."""
  if isinstance(space, spaces.Discrete):
    return specs.DiscreteArray(num_values=int(space.n), dtype=space.dtype, name=name)

  elif isinstance(space, spaces.Box):
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=space.low,
        maximum=space.high,
        name=name)

  elif isinstance(space, spaces.MultiBinary):
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=0.0,
        maximum=1.0,
        name=name)

  elif isinstance(space, spaces.MultiDiscrete):
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=np.zeros(space.shape),
        maximum=space.nvec - 1,
        name=name,
    )

  elif isinstance(space, spaces.Tuple):
    return tuple(_convert_to_spec(s, name) for s in space.spaces)

  elif isinstance(space, spaces.Dict):
    return {
        key: _convert_to_spec(value, key)
        for key, value in space.spaces.items()
    }

  else:
    raise ValueError(f'Unexpected gymnasium space: {space}')
