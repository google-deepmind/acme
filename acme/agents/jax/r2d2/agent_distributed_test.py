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

"""Integration test for the distributed agent."""

import functools
from absl.testing import absltest
import acme
from acme.agents.jax import r2d2
from acme.testing import fakes
import launchpad as lp


class DistributedAgentTest(absltest.TestCase):
  """Simple integration/smoke test for the distributed agent."""

  def test_agent(self):

    env_factory = lambda x: fakes.fake_atari_wrapped(oar_wrapper=True)

    config = r2d2.R2D2Config(
        batch_size=1,
        trace_length=5,
        sequence_period=1,
        samples_per_insert=1.,
        min_replay_size=32,
        burn_in_length=1,
        prefetch_size=2,
        target_update_period=2500,
        max_replay_size=100_000,
        importance_sampling_exponent=0.6,
        priority_exponent=0.9,
        max_priority_weight=0.9,
        bootstrap_n=5,
        clip_rewards=False,
        variable_update_period=400)

    agent = r2d2.DistributedR2D2FromConfig(
        environment_factory=env_factory,
        environment_spec=acme.make_environment_spec(env_factory(False)),
        network_factory=functools.partial(r2d2.make_atari_networks,
                                          config.batch_size),
        config=config,
        seed=0,
        num_actors=1,
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
