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

"""Tests for the distributional MPO agent."""

from typing import Dict, Sequence

from absl.testing import absltest
import acme
from acme import specs
from acme.agents.tf import dmpo
from acme.testing import fakes
from acme.tf import networks
import numpy as np
import sonnet as snt


def make_networks(
    action_spec: specs.Array,
    policy_layer_sizes: Sequence[int] = (300, 200),
    critic_layer_sizes: Sequence[int] = (400, 300),
) -> Dict[str, snt.Module]:
  """Creates networks used by the agent."""

  num_dimensions = np.prod(action_spec.shape, dtype=int)
  critic_layer_sizes = list(critic_layer_sizes)

  policy_network = snt.Sequential([
      networks.LayerNormMLP(policy_layer_sizes),
      networks.MultivariateNormalDiagHead(num_dimensions),
  ])
  # The multiplexer concatenates the (maybe transformed) observations/actions.
  critic_network = snt.Sequential([
      networks.CriticMultiplexer(
          critic_network=networks.LayerNormMLP(critic_layer_sizes)),
      networks.DiscreteValuedHead(0., 1., 10),
  ])

  return {
      'policy': policy_network,
      'critic': critic_network,
  }


class DMPOTest(absltest.TestCase):

  @staticmethod
  def test_dmpo():
    # Create a fake environment to test with.
    environment = fakes.ContinuousEnvironment(episode_length=10)
    spec = specs.make_environment_spec(environment)

    # Create networks.
    agent_networks = make_networks(spec.actions)

    # Construct the agent.
    agent = dmpo.DistributionalMPO(
        spec,
        policy_network=agent_networks['policy'],
        critic_network=agent_networks['critic'],
        batch_size=10,
        samples_per_insert=2,
        min_replay_size=10)

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent)
    loop.run(num_episodes=2)


if __name__ == '__main__':
  absltest.main()
