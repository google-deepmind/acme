# Copyright 2026 DeepMind Technologies Limited. All rights reserved.
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

"""Tests for the DQN builder."""

from absl.testing import absltest
from acme.agents.jax.dqn import builder
from acme.agents.jax.dqn import config


class BuilderTest(absltest.TestCase):

  def test_zero_eval_epsilon_is_preserved(self):
    dqn_builder = builder.DQNBuilder(
        config.DQNConfig(epsilon=0.1, eval_epsilon=0.0), loss_fn=None
    )

    self.assertEqual(dqn_builder._policy_epsilons(evaluation=True), (0.0,))


if __name__ == '__main__':
  absltest.main()
