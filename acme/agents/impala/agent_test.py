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

"""Tests for IMPALA agent."""

from absl.testing import absltest

import acme
from acme import networks
from acme import specs
from acme.agents import impala
from acme.testing import fakes

import numpy as np
import sonnet as snt


def _make_network(action_spec: specs.DiscreteArray) -> snt.RNNCore:
  return snt.DeepRNN([
      snt.Flatten(),
      snt.LSTM(20),
      snt.nets.MLP([50, 50]),
      networks.PolicyValueHead(action_spec.num_values),
  ])


class IMPALATest(absltest.TestCase):

  def test_impala(self):
    # Create a fake environment to test with.
    environment = fakes.DiscreteEnvironment(
        num_actions=5,
        num_observations=10,
        obs_dtype=np.float32,
        episode_length=10)
    spec = specs.make_environment_spec(environment)

    # Construct the agent.
    agent = impala.IMPALA(
        environment_spec=spec,
        network=_make_network(spec.actions),
        sequence_length=3,
        sequence_period=3,
        batch_size=6,
    )

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent)
    loop.run(num_episodes=20)


if __name__ == '__main__':
  absltest.main()
