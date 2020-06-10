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
from acme.agents.tf import dqn
from acme.testing import fakes

import numpy as np
import sonnet as snt


def _make_network(action_spec: specs.DiscreteArray) -> snt.Module:
  return snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([50, 50, action_spec.num_values]),
  ])


class DQNTest(absltest.TestCase):

  def test_dqn(self):
    # Create a fake environment to test with.
    environment = fakes.DiscreteEnvironment(
        num_actions=5,
        num_observations=10,
        obs_dtype=np.float32,
        episode_length=10)
    spec = specs.make_environment_spec(environment)

    # Construct the agent.
    agent = dqn.DQN(
        environment_spec=spec,
        network=_make_network(spec.actions),
        batch_size=10,
        samples_per_insert=2,
        min_replay_size=10)

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent)
    loop.run(num_episodes=2)


if __name__ == '__main__':
  absltest.main()
