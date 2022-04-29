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

"""Tests for core.py."""

from typing import List

from acme import core
from acme import types

from absl.testing import absltest


class StepCountingLearner(core.Learner):
  """A learner which counts `num_steps` and then raises `StopIteration`."""

  def __init__(self, num_steps: int):
    self.step_count = 0
    self.num_steps = num_steps

  def step(self):
    self.step_count += 1
    if self.step_count >= self.num_steps:
      raise StopIteration()

  def get_variables(self, unused: List[str]) -> List[types.NestedArray]:
    del unused
    return []


class CoreTest(absltest.TestCase):

  def test_learner_run_with_limit(self):
    learner = StepCountingLearner(100)
    learner.run(7)
    self.assertEqual(learner.step_count, 7)

  def test_learner_run_no_limit(self):
    learner = StepCountingLearner(100)
    with self.assertRaises(StopIteration):
      learner.run()
    self.assertEqual(learner.step_count, 100)


if __name__ == '__main__':
  absltest.main()
