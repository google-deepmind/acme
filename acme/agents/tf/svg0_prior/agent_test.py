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

"""Tests for the SVG agent."""

import sys
from typing import Dict, Sequence

import acme
from acme import specs
from acme import types
from acme.agents.tf import svg0_prior
from acme.testing import fakes
from acme.tf import networks
from acme.tf import utils as tf2_utils
import numpy as np
import sonnet as snt

from absl.testing import absltest


def make_networks(
    action_spec: types.NestedSpec,
    policy_layer_sizes: Sequence[int] = (10, 10),
    critic_layer_sizes: Sequence[int] = (10, 10),
) -> Dict[str, snt.Module]:
  """Creates networks used by the agent."""
  # Get total number of action dimensions from action spec.
  num_dimensions = np.prod(action_spec.shape, dtype=int)

  policy_network = snt.Sequential([
      tf2_utils.batch_concat,
      networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
      networks.MultivariateNormalDiagHead(
          num_dimensions,
          tanh_mean=True,
          min_scale=0.3,
          init_scale=0.7,
          fixed_scale=False,
          use_tfd_independent=False)
  ])
  # The multiplexer concatenates the (maybe transformed) observations/actions.
  multiplexer = networks.CriticMultiplexer()
  critic_network = snt.Sequential([
      multiplexer,
      networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
      networks.NearZeroInitializedLinear(1),
  ])

  return {
      'policy': policy_network,
      'critic': critic_network,
  }


class SVG0Test(absltest.TestCase):

  def test_svg0(self):
    # Create a fake environment to test with.
    environment = fakes.ContinuousEnvironment(episode_length=10)
    spec = specs.make_environment_spec(environment)

    # Create the networks.
    agent_networks = make_networks(spec.actions)

    # Construct the agent.
    agent = svg0_prior.SVG0(
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

    # Imports check


if __name__ == '__main__':
  absltest.main()
