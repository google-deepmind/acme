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

"""Multiagent utilities."""

from acme import specs
from acme.multiagent import types
import dm_env


def get_agent_spec(env_spec: specs.EnvironmentSpec,
                   agent_id: types.AgentID) -> specs.EnvironmentSpec:
  """Returns a single agent spec from environment spec.

  Args:
    env_spec: environment spec, wherein observation, action, and reward specs
      are simply lists (with each entry specifying the respective spec for the
      given agent index). Discounts are scalars shared amongst agents.
    agent_id: agent index.
  """
  return specs.EnvironmentSpec(
      actions=env_spec.actions[agent_id],
      discounts=env_spec.discounts,
      observations=env_spec.observations[agent_id],
      rewards=env_spec.rewards[agent_id])


def get_agent_timestep(timestep: dm_env.TimeStep,
                       agent_id: types.AgentID) -> dm_env.TimeStep:
  """Returns the extracted timestep for a particular agent."""
  # Discounts are assumed to be shared amongst agents
  reward = None if timestep.reward is None else timestep.reward[agent_id]
  return dm_env.TimeStep(
      observation=timestep.observation[agent_id],
      reward=reward,
      discount=timestep.discount,
      step_type=timestep.step_type)
