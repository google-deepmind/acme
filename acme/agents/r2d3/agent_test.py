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

"""Tests for R2D3 agent."""

from absl.testing import absltest

import acme
from acme import networks
from acme import specs
from acme.agents import r2d3
from acme.agents.dqfd import bsuite_demonstrations
from acme.testing import fakes

import dm_env
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


class R2D3Test(absltest.TestCase):

  def test_r2d3(self):
    # Create a fake environment to test with.
    environment = fakes.DiscreteEnvironment(
        num_actions=5,
        num_observations=10,
        obs_dtype=np.float32,
        episode_length=10)
    spec = specs.make_environment_spec(environment)

    # Build demonstrations.
    dummy_action = np.zeros((), dtype=np.int32)
    recorder = bsuite_demonstrations.DemonstrationRecorder()
    timestep = environment.reset()
    while timestep.step_type is not dm_env.StepType.LAST:
      recorder.step(timestep, dummy_action)
      timestep = environment.step(dummy_action)
    recorder.step(timestep, dummy_action)
    recorder.record_episode()

    # Construct the agent.
    agent = r2d3.R2D3(
        environment_spec=spec,
        network=SimpleNetwork(spec.actions),
        target_network=SimpleNetwork(spec.actions),
        demonstration_dataset=recorder.make_tf_dataset(),
        demonstration_ratio=0.5,
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
