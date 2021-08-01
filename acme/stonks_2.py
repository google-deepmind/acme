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

"""Tests for DQN agent."""

from absl.testing import absltest
import acme
from acme import specs
from acme.agents.jax import dqn
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.testing import fakes
import haiku as hk
import numpy as np

import ray

@ray.remote
class DQNTest():

  def test_dqn(self):
    # Create a fake environment to test with.
    environment = fakes.DiscreteEnvironment(
        num_actions=5,
        num_observations=10,
        obs_shape=(10, 5),
        obs_dtype=np.float32,
        episode_length=10)
    spec = specs.make_environment_spec(environment)

    def network(x):
      model = hk.Sequential([
          hk.Flatten(),
          hk.nets.MLP([50, 50, spec.actions.num_values])
      ])
      return model(x)

    # Make network purely functional
    network_hk = hk.without_apply_rng(hk.transform(network, apply_rng=True))
    dummy_obs = utils.add_batch_dim(utils.zeros_like(spec.observations))

    network = networks_lib.FeedForwardNetwork(
        init=lambda rng: network_hk.init(rng, dummy_obs),
        apply=network_hk.apply)

    # Construct the agent.
    agent = dqn.DQN(
        environment_spec=spec,
        network=network,
        batch_size=10,
        samples_per_insert=2,
        min_replay_size=10)

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent)
    print("running loop")
    loop.run(num_episodes=20)


if __name__ == '__main__':
  t = DQNTest.remote()
  ray.get(t.test_dqn.remote())
