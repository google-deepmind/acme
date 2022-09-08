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

"""Tests for the step limit wrapper."""

from acme import wrappers
from acme.testing import fakes
import numpy as np

from absl.testing import absltest

ACTION = np.array(0, dtype=np.int32)


class StepLimitWrapperTest(absltest.TestCase):

  def test_step(self):
    fake_env = fakes.DiscreteEnvironment(episode_length=5)
    env = wrappers.StepLimitWrapper(fake_env, step_limit=2)

    env.reset()
    env.step(ACTION)
    self.assertTrue(env.step(ACTION).last())

  def test_step_on_new_env(self):
    fake_env = fakes.DiscreteEnvironment(episode_length=5)
    env = wrappers.StepLimitWrapper(fake_env, step_limit=2)

    self.assertTrue(env.step(ACTION).first())
    self.assertFalse(env.step(ACTION).last())
    self.assertTrue(env.step(ACTION).last())

  def test_step_after_truncation(self):
    fake_env = fakes.DiscreteEnvironment(episode_length=5)
    env = wrappers.StepLimitWrapper(fake_env, step_limit=2)

    env.reset()
    env.step(ACTION)
    self.assertTrue(env.step(ACTION).last())

    self.assertTrue(env.step(ACTION).first())
    self.assertFalse(env.step(ACTION).last())
    self.assertTrue(env.step(ACTION).last())

  def test_step_after_termination(self):
    fake_env = fakes.DiscreteEnvironment(episode_length=5)

    fake_env.reset()
    fake_env.step(ACTION)
    fake_env.step(ACTION)
    fake_env.step(ACTION)
    fake_env.step(ACTION)
    self.assertTrue(fake_env.step(ACTION).last())

    env = wrappers.StepLimitWrapper(fake_env, step_limit=2)

    self.assertTrue(env.step(ACTION).first())
    self.assertFalse(env.step(ACTION).last())
    self.assertTrue(env.step(ACTION).last())


if __name__ == '__main__':
  absltest.main()
