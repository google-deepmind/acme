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

"""Tests for search.py."""

from typing import Text

from absl.testing import absltest
from absl.testing import parameterized

from acme.agents.tf.mcts import search
from acme.agents.tf.mcts.models import simulator

from bsuite.environments import catch
import numpy as np


class TestSearch(parameterized.TestCase):

  @parameterized.parameters([
      'puct',
      'bfs',
  ])
  def test_catch(self, policy_type: Text):
    env = catch.Catch(rows=2)
    num_actions = env.action_spec().num_values
    model = simulator.Simulator(env)
    eval_fn = lambda _: (np.ones(num_actions) / num_actions, 0.)

    timestep = env.reset()
    model.reset()

    search_policy = search.bfs if policy_type == 'bfs' else search.puct

    root = search.mcts(
        observation=timestep.observation,
        model=model,
        search_policy=search_policy,
        evaluation=eval_fn,
        num_simulations=100,
        num_actions=num_actions)

    values = np.array([c.value for c in root.children.values()])
    best_action = search.argmax(values)

    if env._paddle_x > env._ball_x:
      self.assertEqual(best_action, 0)
    if env._paddle_x == env._ball_x:
      self.assertEqual(best_action, 1)
    if env._paddle_x < env._ball_x:
      self.assertEqual(best_action, 2)


if __name__ == '__main__':
  absltest.main()
