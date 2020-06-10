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

"""Tests for the DDPG agent."""

from typing import Dict, Sequence

from absl.testing import absltest
import acme
from acme import specs
from acme import types
from acme.agents.tf import ddpg
from acme.testing import fakes
from acme.tf import networks
import numpy as np
import sonnet as snt
import tensorflow as tf


def make_networks(
    action_spec: types.NestedSpec,
    policy_layer_sizes: Sequence[int] = (10, 10),
    critic_layer_sizes: Sequence[int] = (10, 10),
) -> Dict[str, snt.Module]:
  """Creates networks used by the agent."""

  num_dimensions = np.prod(action_spec.shape, dtype=int)
  policy_layer_sizes = list(policy_layer_sizes) + [num_dimensions]
  critic_layer_sizes = list(critic_layer_sizes) + [1]

  policy_network = snt.Sequential(
      [networks.LayerNormMLP(policy_layer_sizes), tf.tanh])
  # The multiplexer concatenates the (maybe transformed) observations/actions.
  critic_network = networks.CriticMultiplexer(
      critic_network=networks.LayerNormMLP(critic_layer_sizes))

  return {
      'policy': policy_network,
      'critic': critic_network,
  }


class DDPGTest(absltest.TestCase):

  def test_ddpg(self):
    # Create a fake environment to test with.
    environment = fakes.ContinuousEnvironment(episode_length=10, bounded=True)
    spec = specs.make_environment_spec(environment)

    # Create the networks to optimize (online) and target networks.
    agent_networks = make_networks(spec.actions)

    # Construct the agent.
    agent = ddpg.DDPG(
        environment_spec=spec,
        policy_network=agent_networks['policy'],
        critic_network=agent_networks['critic'],
        batch_size=10,
        samples_per_insert=2,
        min_replay_size=10,
    )

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent)
    loop.run(num_episodes=2)


if __name__ == '__main__':
  absltest.main()
