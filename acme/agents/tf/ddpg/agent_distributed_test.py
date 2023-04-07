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

import launchpad as lp
import numpy as np
import sonnet as snt
from absl.testing import absltest

import acme
from acme import specs
from acme.agents.tf import ddpg
from acme.testing import fakes
from acme.tf import networks
from acme.tf import utils as tf2_utils


def make_networks(action_spec: specs.BoundedArray):
    """Creates simple networks for testing.."""

    num_dimensions = np.prod(action_spec.shape, dtype=int)

    # Create the observation network shared between the policy and critic.
    observation_network = tf2_utils.batch_concat

    # Create the policy network (head) and the evaluation network.
    policy_network = snt.Sequential(
        [
            networks.LayerNormMLP([50], activate_final=True),
            networks.NearZeroInitializedLinear(num_dimensions),
            networks.TanhToSpec(action_spec),
        ]
    )
    evaluator_network = snt.Sequential([observation_network, policy_network])

    # Create the critic network.
    critic_network = snt.Sequential(
        [
            # The multiplexer concatenates the observations/actions.
            networks.CriticMultiplexer(),
            networks.LayerNormMLP([50], activate_final=True),
            networks.NearZeroInitializedLinear(1),
        ]
    )

    return {
        "policy": policy_network,
        "critic": critic_network,
        "observation": observation_network,
        "evaluator": evaluator_network,
    }


class DistributedAgentTest(absltest.TestCase):
    """Simple integration/smoke test for the distributed agent."""

    def test_agent(self):

        agent = ddpg.DistributedDDPG(
            environment_factory=lambda x: fakes.ContinuousEnvironment(bounded=True),
            network_factory=make_networks,
            num_actors=2,
            batch_size=32,
            min_replay_size=32,
            max_replay_size=1000,
        )
        program = agent.build()

        (learner_node,) = program.groups["learner"]
        learner_node.disable_run()

        lp.launch(program, launch_type="test_mt")

        learner: acme.Learner = learner_node.create_handle().dereference()

        for _ in range(5):
            learner.step()


if __name__ == "__main__":
    absltest.main()
