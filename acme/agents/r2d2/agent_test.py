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

"""Tests for RDQN agent."""

from absl.testing import absltest

import acme
from acme import networks
from acme import specs
from acme.agents import r2d2
from acme.testing import fakes

import numpy as np
import sonnet as snt


class SimpleNetwork(networks.RNNCore):

  def __init__(self, action_spec: specs.DiscreteArray):
    super().__init__(name='r2d2_test_network')
    self._net = snt.DeepRNN([
        snt.Flatten(),
        snt.LSTM(20),
        snt.nets.MLP([50, 50, action_spec.num_values])
    ])

  def __call__(self, inputs, state):
    return self._net(inputs, state)

  def initial_state(self, batch_size: int, **kwargs):
    return self._net.initial_state(batch_size)

  def unroll(self, inputs, state, sequence_length):
    return snt.static_unroll(self._net, inputs, state, sequence_length)


class R2D2Test(absltest.TestCase):

  def test_r2d2(self):
    # Create a fake environment to test with.
    # TODO(b/152596848): Allow R2D2 to deal with integer observations.
    environment = fakes.DiscreteEnvironment(
        num_actions=5,
        num_observations=10,
        obs_shape=(10, 4),
        obs_dtype=np.float32,
        episode_length=10)
    spec = specs.make_environment_spec(environment)

    # Construct the agent.
    agent = r2d2.R2D2(
        environment_spec=spec,
        network=SimpleNetwork(spec.actions),
        batch_size=10,
        samples_per_insert=2,
        min_replay_size=10,
        burn_in_length=2,
        trace_length=6,
        replay_period=4,
        checkpoint=False,
    )

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent)
    loop.run(num_episodes=5)


if __name__ == '__main__':
  absltest.main()
