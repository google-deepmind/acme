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
from acme.agents.jax import ars
from acme.testing import fakes
import launchpad as lp

from absl.testing import absltest


class DistributedAgentTest(absltest.TestCase):
  """Simple integration/smoke test for the distributed agent."""

  def test_agent(self):
    env_factory = lambda seed: fakes.ContinuousEnvironment(  # pylint: disable=g-long-lambda
        episode_length=100,
        action_dim=3,
        observation_dim=17,
        bounded=True)
    config = ars.ARSConfig(num_directions=3, top_directions=2)

    agent = ars.DistributedARS(
        environment_factory=env_factory,
        network_factory=ars.make_networks,
        config=config,
        seed=0,
        num_actors=2,
    )

    program = agent.build()
    (learner_node,) = program.groups['learner']
    learner_node.disable_run()  # pytype: disable=attribute-error

    lp.launch(program, launch_type='test_mt')

    learner: acme.Learner = learner_node.create_handle().dereference()

    for _ in range(5):
      learner.step()


if __name__ == '__main__':
  absltest.main()
