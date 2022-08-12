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

"""Multiagent dict-indexed environment wrapped."""

from typing import Any, Dict, List, TypeVar, Union
from acme import types

from acme.wrappers import base
import dm_env

V = TypeVar('V')


class MultiagentDictKeyWrapper(base.EnvironmentWrapper):
  """Wrapper that converts list-indexed multiagent environments to dict-indexed.

  Specifically, if the underlying environment observation and actions are:
    observation = [observation_agent_0, observation_agent_1, ...]
    action = [action_agent_0, action_agent_1, ...]

  They are converted instead to:
    observation = {'0': observation_agent_0, '1': observation_agent_1, ...}
    action = {'0': action_agent_0, '1': action_agent_1, ...}

  This can be helpful in situations where dict-based structures are natively
  supported, whereas lists are not (e.g., in tfds, where ragged observation data
  can directly be supported if dicts, but not natively supported as lists).
  """

  def __init__(self, environment: dm_env.Environment):
    self._environment = environment
    # Convert action and observation specs.
    self._action_spec = self._list_to_dict(self._environment.action_spec())
    self._discount_spec = self._list_to_dict(self._environment.discount_spec())
    self._observation_spec = self._list_to_dict(
        self._environment.observation_spec())
    self._reward_spec = self._list_to_dict(self._environment.reward_spec())

  def _list_to_dict(self, data: Union[List[V], V]) -> Union[Dict[str, V], V]:
    """Convert list-indexed data to dict-indexed, otherwise passthrough."""
    if isinstance(data, list):
      return {str(k): v for k, v in enumerate(data)}
    return data

  def _dict_to_list(self, data: Union[Dict[str, V], V]) -> Union[List[V], V]:
    """Convert dict-indexed data to list-indexed, otherwise passthrough."""
    if isinstance(data, dict):
      return [data[str(i_agent)]
              for i_agent in range(self._environment.num_agents)]   # pytype: disable=attribute-error
    return data

  def _convert_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    return timestep._replace(
        reward=self._list_to_dict(timestep.reward),
        discount=self._list_to_dict(timestep.discount),
        observation=self._list_to_dict(timestep.observation))

  def step(self, action: Dict[int, Any]) -> dm_env.TimeStep:
    return self._convert_timestep(
        self._environment.step(self._dict_to_list(action)))

  def reset(self) -> dm_env.TimeStep:
    return self._convert_timestep(self._environment.reset())

  def action_spec(self) -> types.NestedSpec:  # Internal pytype check.
    return self._action_spec

  def discount_spec(self) -> types.NestedSpec:  # Internal pytype check.
    return self._discount_spec

  def observation_spec(self) -> types.NestedSpec:  # Internal pytype check.
    return self._observation_spec

  def reward_spec(self) -> types.NestedSpec:  # Internal pytype check.
    return self._reward_spec
