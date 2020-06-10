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

"""Tests for the MCTS agent."""

from absl.testing import absltest
import acme
from acme import specs
from acme.agents.tf import mcts
from acme.agents.tf.mcts.models import simulator
from acme.testing import fakes
from acme.tf import networks
import numpy as np
import sonnet as snt


class MCTSTest(absltest.TestCase):

  def test_mcts(self):
    # Create a fake environment to test with.
    num_actions = 5
    environment = fakes.DiscreteEnvironment(
        num_actions=num_actions,
        num_observations=10,
        obs_dtype=np.float32,
        episode_length=10)
    spec = specs.make_environment_spec(environment)

    network = snt.Sequential([
        snt.Flatten(),
        snt.nets.MLP([50, 50]),
        networks.PolicyValueHead(spec.actions.num_values),
    ])
    model = simulator.Simulator(environment)
    optimizer = snt.optimizers.Adam(1e-3)

    # Construct the agent.
    agent = mcts.MCTS(
        environment_spec=spec,
        network=network,
        model=model,
        optimizer=optimizer,
        n_step=1,
        discount=1.,
        replay_capacity=100,
        num_simulations=10,
        batch_size=10)

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent)
    loop.run(num_episodes=2)


if __name__ == '__main__':
  absltest.main()
