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

"""Tests for the DQN builder."""

from absl.testing import absltest
from acme.agents.jax.dqn import builder
from acme.agents.jax.dqn import config as dqn_config


class PolicyEpsilonsTest(absltest.TestCase):

  def _make_builder(self, **config_kwargs) -> builder.DQNBuilder:
    config = dqn_config.DQNConfig(**config_kwargs)
    return builder.DQNBuilder(config=config, loss_fn=None)

  def test_explicit_zero_eval_epsilon_is_honored(self):
    """0.0 is a valid eval epsilon (greedy eval), not "unset"."""
    b = self._make_builder(epsilon=0.05, eval_epsilon=0.0)
    self.assertEqual(b._policy_epsilons(evaluation=True), (0.0,))

  def test_unset_eval_epsilon_falls_back_to_behavior_epsilon(self):
    b = self._make_builder(epsilon=0.05, eval_epsilon=None)
    self.assertEqual(b._policy_epsilons(evaluation=True), (0.05,))

  def test_nonzero_eval_epsilon_is_honored(self):
    b = self._make_builder(epsilon=0.05, eval_epsilon=0.1)
    self.assertEqual(b._policy_epsilons(evaluation=True), (0.1,))

  def test_eval_epsilon_is_ignored_when_not_evaluating(self):
    b = self._make_builder(epsilon=0.05, eval_epsilon=0.0)
    self.assertEqual(b._policy_epsilons(evaluation=False), (0.05,))


if __name__ == '__main__':
  absltest.main()
