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

"""Tests for frozen_learner."""

from unittest import mock

import acme
from acme.utils import frozen_learner
from absl.testing import absltest


class FrozenLearnerTest(absltest.TestCase):

  @mock.patch.object(acme, 'Learner', autospec=True)
  def test_step_fn(self, mock_learner):
    num_calls = 0

    def step_fn():
      nonlocal num_calls
      num_calls += 1

    learner = frozen_learner.FrozenLearner(mock_learner, step_fn=step_fn)

    # Step two times.
    learner.step()
    learner.step()

    self.assertEqual(num_calls, 2)
    # step() method of the wrapped learner should not be called.
    mock_learner.step.assert_not_called()

  @mock.patch.object(acme, 'Learner', autospec=True)
  def test_no_step_fn(self, mock_learner):
    learner = frozen_learner.FrozenLearner(mock_learner)
    learner.step()
    # step() method of the wrapped learner should not be called.
    mock_learner.step.assert_not_called()

  @mock.patch.object(acme, 'Learner', autospec=True)
  def test_save_and_restore(self, mock_learner):
    learner = frozen_learner.FrozenLearner(mock_learner)

    mock_learner.save.return_value = 'state1'

    state = learner.save()
    self.assertEqual(state, 'state1')

    learner.restore('state2')
    # State of the wrapped learner should be restored.
    mock_learner.restore.assert_called_once_with('state2')

  @mock.patch.object(acme, 'Learner', autospec=True)
  def test_get_variables(self, mock_learner):
    learner = frozen_learner.FrozenLearner(mock_learner)

    mock_learner.get_variables.return_value = [1, 2]

    variables = learner.get_variables(['a', 'b'])
    # Values should match with those returned by the wrapped learner.
    self.assertEqual(variables, [1, 2])
    mock_learner.get_variables.assert_called_once_with(['a', 'b'])


if __name__ == '__main__':
  absltest.main()
