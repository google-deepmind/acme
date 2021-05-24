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
from absl.testing import parameterized
from acme.wrappers import atari_wrapper
from dm_env import specs
import numpy as np

SKIP_GYM_TESTS = False
SKIP_GYM_MESSAGE = 'gym not installed.'
SKIP_ATARI_TESTS = False
SKIP_ATARI_MESSAGE = ''

try:
  # pylint: disable=g-import-not-at-top
  from acme.wrappers import gym_wrapper
  import gym
  # pylint: enable=g-import-not-at-top
except ModuleNotFoundError:
  SKIP_GYM_TESTS = True


try:
  import atari_py  # pylint: disable=g-import-not-at-top
  atari_py.get_game_path('pong')
except ModuleNotFoundError as e:
  SKIP_ATARI_TESTS = True
  SKIP_ATARI_MESSAGE = str(e)
except Exception as e:  # pylint: disable=broad-except
  # This exception is raised by atari_py.get_game_path('pong') if the Atari ROM
  # file has not been installed.
  SKIP_ATARI_TESTS = True
  SKIP_ATARI_MESSAGE = str(e)
  del atari_py
else:
  del atari_py


@unittest.skipIf(SKIP_ATARI_TESTS, SKIP_ATARI_MESSAGE)
@unittest.skipIf(SKIP_GYM_TESTS, SKIP_GYM_MESSAGE)
class AtariWrapperTest(parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_pong(self, zero_discount_on_life_loss: bool):
    env = gym.make('PongNoFrameskip-v4', full_action_space=True)
    env = gym_wrapper.GymAtariAdapter(env)
    env = atari_wrapper.AtariWrapper(
        env, zero_discount_on_life_loss=zero_discount_on_life_loss)

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

    # Check that the `render` call gets delegated to the underlying Gym env.
    env.render('rgb_array')

    # Test step.
    timestep = env.reset()
    self.assertTrue(timestep.first())
    _ = env.step(0)
    env.close()


if __name__ == '__main__':
  absltest.main()
