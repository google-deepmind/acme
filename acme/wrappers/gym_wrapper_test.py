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

"""Tests for gym_wrapper."""

import unittest
from absl.testing import absltest
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
class GymWrapperTest(absltest.TestCase):

  def test_gym_cartpole(self):
    env = gym_wrapper.GymWrapper(gym.make('CartPole-v0'))

    # Test converted observation spec.
    observation_spec: specs.BoundedArray = env.observation_spec()
    self.assertEqual(type(observation_spec), specs.BoundedArray)
    self.assertEqual(observation_spec.shape, (4,))
    self.assertEqual(observation_spec.minimum.shape, (4,))
    self.assertEqual(observation_spec.maximum.shape, (4,))
    self.assertEqual(observation_spec.dtype, np.dtype('float32'))

    # Test converted action spec.
    action_spec: specs.BoundedArray = env.action_spec()
    self.assertEqual(type(action_spec), specs.DiscreteArray)
    self.assertEqual(action_spec.shape, ())
    self.assertEqual(action_spec.minimum, 0)
    self.assertEqual(action_spec.maximum, 1)
    self.assertEqual(action_spec.num_values, 2)
    self.assertEqual(action_spec.dtype, np.dtype('int64'))

    # Test step.
    timestep = env.reset()
    self.assertTrue(timestep.first())
    timestep = env.step(1)
    self.assertEqual(timestep.reward, 1.0)
    self.assertEqual(timestep.observation.shape, (4,))
    env.close()

  def test_early_truncation(self):
    # Pendulum has no early termination condition.
    gym_env = gym.make('Pendulum-v0')
    env = gym_wrapper.GymWrapper(gym_env)
    ts = env.reset()
    while not ts.last():
      ts = env.step(env.action_spec().generate_value())
    self.assertEqual(ts.discount, 1.0)
    env.close()


@unittest.skipIf(SKIP_GYM_TESTS, SKIP_GYM_MESSAGE)
class AtariGymWrapperTest(absltest.TestCase):

  def test_pong(self):
    env = gym.make('PongNoFrameskip-v4', full_action_space=True)
    env = gym_wrapper.GymAtariAdapter(env)

    # Test converted observation spec. This should expose (RGB, LIVES).
    observation_spec = env.observation_spec()
    self.assertEqual(type(observation_spec[0]), specs.BoundedArray)
    self.assertEqual(type(observation_spec[1]), specs.Array)

    # Test converted action spec.
    action_spec: specs.DiscreteArray = env.action_spec()[0]
    self.assertEqual(type(action_spec), specs.DiscreteArray)
    self.assertEqual(action_spec.shape, ())
    self.assertEqual(action_spec.minimum, 0)
    self.assertEqual(action_spec.maximum, 17)
    self.assertEqual(action_spec.num_values, 18)
    self.assertEqual(action_spec.dtype, np.dtype('int64'))

    # Test step.
    timestep = env.reset()
    self.assertTrue(timestep.first())
    _ = env.step([np.array(0)])
    env.close()


if __name__ == '__main__':
  absltest.main()
