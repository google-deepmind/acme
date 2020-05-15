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

"""Tests for atari_wrapper."""

import unittest
from absl.testing import absltest
from acme.wrappers import atari_wrapper
from dm_env import specs
import numpy as np

SKIP_GYM_TESTS = False
SKIP_GYM_MESSAGE = 'gym not installed.'

try:
  # pylint: disable=g-import-not-at-top
  from acme.wrappers import gym_wrapper
  import gym
except ModuleNotFoundError:
  SKIP_GYM_TESTS = True


@unittest.skipIf(SKIP_GYM_TESTS, SKIP_GYM_MESSAGE)
class AtariWrapperTest(absltest.TestCase):

  def test_pong(self):
    env = gym.make('PongNoFrameskip-v4', full_action_space=True)
    env = gym_wrapper.GymAtariAdapter(env)
    env = atari_wrapper.AtariWrapper(env)

    # Test converted observation spec.
    observation_spec = env.observation_spec()
    self.assertEqual(type(observation_spec), specs.Array)

    # Test converted action spec.
    action_spec: specs.DiscreteArray = env.action_spec()
    self.assertEqual(type(action_spec), specs.DiscreteArray)
    self.assertEqual(action_spec.shape, ())
    self.assertEqual(action_spec.minimum, 0)
    self.assertEqual(action_spec.maximum, 17)
    self.assertEqual(action_spec.num_values, 18)
    self.assertEqual(action_spec.dtype, np.dtype('int32'))

    # Test step.
    timestep = env.reset()
    self.assertTrue(timestep.first())
    _ = env.step(0)
    env.close()


if __name__ == '__main__':
  absltest.main()
