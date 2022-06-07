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

import acme
from acme.agents.tf import impala
from acme.testing import fakes
from acme.tf import networks
import launchpad as lp

from absl.testing import absltest


class DistributedAgentTest(absltest.TestCase):
  """Simple integration/smoke test for the distributed agent."""

  def test_atari(self):
    """Tests that the agent can run for some steps without crashing."""
    env_factory = lambda x: fakes.fake_atari_wrapped(oar_wrapper=True)
    net_factory = lambda spec: networks.IMPALAAtariNetwork(spec.num_values)

    agent = impala.DistributedIMPALA(
        environment_factory=env_factory,
        network_factory=net_factory,
        num_actors=2,
        batch_size=32,
        sequence_length=5,
        sequence_period=1,
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
