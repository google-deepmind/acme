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

"""Tests for the TD3 agent."""

from absl.testing import absltest
import acme
from acme import specs
from acme.agents.jax import td3
from acme.testing import fakes
from acme.utils import counting


class TD3Test(absltest.TestCase):

  def test_td3(self):
    # Create a fake environment to test with.
    environment = fakes.ContinuousEnvironment(
        episode_length=10, action_dim=3, observation_dim=5, bounded=True)
    spec = specs.make_environment_spec(environment)

    # Create the networks.
    network = td3.make_networks(spec)

    config = td3.TD3Config(
        batch_size=10,
        min_replay_size=1)

    counter = counting.Counter()
    agent = td3.TD3(spec=spec, network=network, config=config, seed=0,
                    counter=counter)

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent, counter=counter)
    loop.run(num_episodes=2)


if __name__ == '__main__':
  absltest.main()
