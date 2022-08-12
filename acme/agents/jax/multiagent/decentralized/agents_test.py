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

"""Tests for agents."""

import acme
from acme.agents.jax.multiagent.decentralized import agents
from acme.agents.jax.multiagent.decentralized import factories
from acme.testing import fakes
from acme.testing import multiagent_fakes
from absl.testing import absltest


class AgentsTest(absltest.TestCase):

  def test_init_decentralized_multiagent(self):
    batch_size = 5
    agent_indices = ['a', '99', 'Z']
    environment_spec = multiagent_fakes.make_multiagent_environment_spec(
        agent_indices)
    env = fakes.Environment(environment_spec, episode_length=4)
    agent_types = {
        agent_id: factories.DefaultSupportedAgent.TD3
        for agent_id in agent_indices
    }
    agt_configs = {'sigma': 0.3, 'target_sigma': 0.3}
    config_overrides = {
        k: agt_configs for k, v in agent_types.items()
        if v == factories.DefaultSupportedAgent.TD3
    }

    agent, _ = agents.init_decentralized_multiagent(
        agent_types=agent_types,
        environment_spec=environment_spec,
        seed=1,
        batch_size=batch_size,
        workdir=None,
        init_network_fn=None,
        save_data=False,
        config_overrides=config_overrides)

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(env, agent)
    loop.run(num_episodes=10)


if __name__ == '__main__':
  absltest.main()
