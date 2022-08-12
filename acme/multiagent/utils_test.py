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

"""Tests for multiagent_utils."""

from acme import specs
from acme.multiagent import utils as multiagent_utils
from acme.testing import fakes
from acme.testing import multiagent_fakes
import dm_env
from absl.testing import absltest


class UtilsTest(absltest.TestCase):

  def test_get_agent_spec(self):
    agent_indices = ['a', '99', 'Z']
    spec = multiagent_fakes.make_multiagent_environment_spec(agent_indices)
    for agent_id in spec.actions.keys():
      single_agent_spec = multiagent_utils.get_agent_spec(
          spec, agent_id=agent_id)
      expected_spec = specs.EnvironmentSpec(
          actions=spec.actions[agent_id],
          discounts=spec.discounts,
          observations=spec.observations[agent_id],
          rewards=spec.rewards[agent_id]
      )
      self.assertEqual(single_agent_spec, expected_spec)

  def test_get_agent_timestep(self):
    agent_indices = ['a', '99', 'Z']
    spec = multiagent_fakes.make_multiagent_environment_spec(agent_indices)
    env = fakes.Environment(spec)
    timestep = env.reset()
    for agent_id in spec.actions.keys():
      single_agent_timestep = multiagent_utils.get_agent_timestep(
          timestep, agent_id)
      expected_timestep = dm_env.TimeStep(
          observation=timestep.observation[agent_id],
          reward=None,
          discount=None,
          step_type=timestep.step_type
      )
      self.assertEqual(single_agent_timestep, expected_timestep)


if __name__ == '__main__':
  absltest.main()
