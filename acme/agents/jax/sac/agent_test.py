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

"""Tests for the SAC agent."""

from absl.testing import absltest
import acme
from acme import specs
from acme.agents.jax.sac import agents
from acme.agents.jax.sac import config as sac_config
from acme.agents.jax.sac import networks
from acme.testing import fakes
from acme.utils import counting


class SACTest(absltest.TestCase):

  def test_sac(self):
    # Create a fake environment to test with.
    environment = fakes.ContinuousEnvironment(
        episode_length=10, action_dim=3, observation_dim=5, bounded=True)
    spec = specs.make_environment_spec(environment)

    # Create the networks.
    network = networks.make_networks(spec)

    batch_size = 10
    config = sac_config.SACConfig(
        batch_size=batch_size,
        target_entropy=sac_config.target_entropy_from_env_spec(spec),
        min_replay_size=1)
    counter = counting.Counter()
    agent = agents.SAC(
        spec=spec, network=network, config=config, seed=0, normalize_input=True,
        counter=counter)

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent, counter=counter)
    loop.run(num_episodes=2)


if __name__ == '__main__':
  absltest.main()
