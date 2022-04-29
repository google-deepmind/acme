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

"""Integration test for the distributed agent."""

from typing import Sequence

import acme
from acme import specs
from acme.agents.tf import svg0_prior
from acme.testing import fakes
from acme.tf import networks
from acme.tf import utils as tf2_utils
import launchpad as lp
import numpy as np
import sonnet as snt

from absl.testing import absltest


def make_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (10, 10),
    critic_layer_sizes: Sequence[int] = (10, 10),
):
  """Simple networks for testing.."""

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


class DistributedAgentTest(absltest.TestCase):
  """Simple integration/smoke test for the distributed agent."""

  def test_control_suite(self):
    """Tests that the agent can run on the control suite without crashing."""

    agent = svg0_prior.DistributedSVG0(
        environment_factory=lambda x: fakes.ContinuousEnvironment(),
        network_factory=make_networks,
        num_actors=2,
        batch_size=32,
        min_replay_size=32,
        max_replay_size=1000,
    )
    program = agent.build()

    (learner_node,) = program.groups['learner']
    learner_node.disable_run()

    lp.launch(program, launch_type='test_mt')

    learner: acme.Learner = learner_node.create_handle().dereference()

    for _ in range(5):
      learner.step()


if __name__ == '__main__':
  absltest.main()
