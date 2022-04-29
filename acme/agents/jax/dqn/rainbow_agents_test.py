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

"""Tests for local and distributed rainbow agents."""

import acme
from acme import specs
from acme.agents.jax.dqn import rainbow_agents
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.testing import fakes
import haiku as hk
import launchpad as lp
import numpy as np

from absl.testing import absltest


def network_factory(
    spec: specs.EnvironmentSpec) -> networks_lib.FeedForwardNetwork:

  def network(x):
    model = hk.Sequential([
        hk.Flatten(),
        hk.nets.MLP([50, 50], activate_final=True),
        networks_lib.DiscreteValued(num_actions=spec.actions.num_values),
    ])
    return model(x)

  network_hk = hk.without_apply_rng(hk.transform(network))
  dummy_obs = utils.add_batch_dim(utils.zeros_like(spec.observations))

  return networks_lib.FeedForwardNetwork(
      init=lambda rng: network_hk.init(rng, dummy_obs), apply=network_hk.apply)


class RainbowDQNTest(absltest.TestCase):

  def test_rainbow(self):
    # Create a fake environment to test with.
    environment = fakes.DiscreteEnvironment(
        num_actions=5,
        num_observations=10,
        obs_shape=(10, 5),
        obs_dtype=np.float32,
        episode_length=10)
    spec = specs.make_environment_spec(environment)

    network = network_factory(spec)
    config = rainbow_agents.RainbowConfig(
        batch_size=10,
        min_replay_size=10,
        samples_per_insert_tolerance_rate=0.2)
    # Construct the agent.
    agent = rainbow_agents.RainbowDQN(spec, network, config, seed=0)

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent)
    loop.run(num_episodes=20)


class DistributedRainbowTest(absltest.TestCase):

  def test_distributed_rainbow(self):

    def env_factory(evaluation):
      del evaluation
      return fakes.DiscreteEnvironment(
          num_actions=5,
          num_observations=10,
          obs_shape=(10, 5),
          obs_dtype=np.float32,
          episode_length=10)

    agent = rainbow_agents.DistributedRainbow(
        environment_factory=env_factory,
        network_factory=network_factory,
        config=rainbow_agents.RainbowConfig(
            batch_size=10,
            min_replay_size=10,
            samples_per_insert_tolerance_rate=0.2),
        seed=0,
        num_actors=2)

    program = agent.build()
    (learner_node,) = program.groups['learner']
    learner_node.disable_run()  # pytype: disable=attribute-error

    lp.launch(program, launch_type='test_mt')

    learner: acme.Learner = learner_node.create_handle().dereference()

    for _ in range(5):
      learner.step()


if __name__ == '__main__':
  absltest.main()
