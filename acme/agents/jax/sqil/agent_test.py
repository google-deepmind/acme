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

"""Tests for the SQIL agent."""

import acme
from acme import specs
from acme.agents.jax import sac
from acme.agents.jax import sqil
from acme.testing import fakes
from acme.utils import counting

from absl.testing import absltest


class SQILTest(absltest.TestCase):

  def test_sqil(self):
    # Create a fake environment to test with.
    environment = fakes.ContinuousEnvironment(episode_length=10,
                                              action_dim=3,
                                              observation_dim=5,
                                              bounded=True)
    spec = specs.make_environment_spec(environment)

    batch_size = 8

    networks = sac.make_networks(spec)
    behavior_policy = sac.apply_policy_and_sample(networks)

    # Construct the agent.
    config = sac.SACConfig(batch_size=batch_size,
                           samples_per_insert_tolerance_rate=2.0,
                           min_replay_size=1)
    builder = sac.SACBuilder(config=config)

    counter = counting.Counter()
    agent = sqil.SQIL(
        spec=spec,
        rl_agent=builder,
        network=networks,
        seed=0,
        batch_size=batch_size,
        make_demonstrations=fakes.transition_iterator(environment),
        policy_network=behavior_policy,
        min_replay_size=1,
        counter=counter)

    # Train the agent
    loop = acme.EnvironmentLoop(environment, agent, counter=counter)
    loop.run(num_episodes=1)


if __name__ == '__main__':
  absltest.main()
