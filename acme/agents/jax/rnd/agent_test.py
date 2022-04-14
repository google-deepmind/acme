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

"""Tests for the RND agent."""

from absl.testing import absltest
import acme
from acme import specs
from acme.agents.jax import sac
from acme.agents.jax.rnd import agents
from acme.agents.jax.rnd import config as rnd_config
from acme.agents.jax.rnd import networks as rnd_networks
from acme.testing import fakes
from acme.utils import counting


class RNDTest(absltest.TestCase):

  def test_rnd(self):
    # Create a fake environment to test with.
    environment = fakes.ContinuousEnvironment(
        episode_length=10, action_dim=3, observation_dim=5, bounded=True)
    spec = specs.make_environment_spec(environment)

    networks = sac.make_networks(spec=spec)
    config = sac.SACConfig(samples_per_insert_tolerance_rate=float('inf'),
                           min_replay_size=1)
    sac_builder = sac.SACBuilder(config=config)
    behavior_policy = sac.apply_policy_and_sample(networks)

    counter = counting.Counter()
    # Construct the agent.
    agent = agents.RND(
        spec=spec,
        rl_agent=sac_builder,
        network=rnd_networks.make_networks(
            spec=spec, direct_rl_networks=networks),
        config=rnd_config.RNDConfig(),
        policy_network=behavior_policy,
        seed=0,
        counter=counter,
    )

    # Train the agent.
    train_loop = acme.EnvironmentLoop(environment, agent, counter=counter)
    train_loop.run(num_episodes=2)


if __name__ == '__main__':
  absltest.main()
