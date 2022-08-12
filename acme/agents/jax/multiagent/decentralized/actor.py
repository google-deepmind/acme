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

"""Decentralized multiagent actor."""

from typing import Dict

from acme import core
from acme.jax import networks
from acme.multiagent import types as ma_types
from acme.multiagent import utils as ma_utils
import dm_env


class SimultaneousActingMultiAgentActor(core.Actor):
  """Simultaneous-move actor (see README.md for expected environment interface)."""

  def __init__(self, actors: Dict[ma_types.AgentID, core.Actor]):
    """Initializer.

    Args:
      actors: a dict specifying sub-actors.
    """
    self._actors = actors

  def select_action(
      self, observation: Dict[ma_types.AgentID, networks.Observation]
  ) -> Dict[ma_types.AgentID, networks.Action]:
    return {
        actor_id: actor.select_action(observation[actor_id])
        for actor_id, actor in self._actors.items()
    }

  def observe_first(self, timestep: dm_env.TimeStep):
    for actor_id, actor in self._actors.items():
      sub_timestep = ma_utils.get_agent_timestep(timestep, actor_id)
      actor.observe_first(sub_timestep)

  def observe(self, actions: Dict[ma_types.AgentID, networks.Action],
              next_timestep: dm_env.TimeStep):
    for actor_id, actor in self._actors.items():
      sub_next_timestep = ma_utils.get_agent_timestep(next_timestep, actor_id)
      actor.observe(actions[actor_id], sub_next_timestep)

  def update(self, wait: bool = False):
    for actor in self._actors.values():
      actor.update(wait=wait)
