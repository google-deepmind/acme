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

import numpy as np
from absl.testing import absltest
from dm_env import specs

SKIP_GYM_TESTS = False
SKIP_GYM_MESSAGE = "gym not installed."
SKIP_ATARI_TESTS = False
SKIP_ATARI_MESSAGE = ""

try:
    # pylint: disable=g-import-not-at-top
    import gym

    from acme.wrappers import gym_wrapper

    # pylint: enable=g-import-not-at-top
except ModuleNotFoundError:
    SKIP_GYM_TESTS = True

try:
    import atari_py  # pylint: disable=g-import-not-at-top

    atari_py.get_game_path("pong")
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


@unittest.skipIf(SKIP_GYM_TESTS, SKIP_GYM_MESSAGE)
class GymWrapperTest(absltest.TestCase):
    def test_gym_cartpole(self):
        env = gym_wrapper.GymWrapper(gym.make("CartPole-v0"))

        # Test converted observation spec.
        observation_spec: specs.BoundedArray = env.observation_spec()
        self.assertEqual(type(observation_spec), specs.BoundedArray)
        self.assertEqual(observation_spec.shape, (4,))
        self.assertEqual(observation_spec.minimum.shape, (4,))
        self.assertEqual(observation_spec.maximum.shape, (4,))
        self.assertEqual(observation_spec.dtype, np.dtype("float32"))

        # Test converted action spec.
        action_spec: specs.BoundedArray = env.action_spec()
        self.assertEqual(type(action_spec), specs.DiscreteArray)
        self.assertEqual(action_spec.shape, ())
        self.assertEqual(action_spec.minimum, 0)
        self.assertEqual(action_spec.maximum, 1)
        self.assertEqual(action_spec.num_values, 2)
        self.assertEqual(action_spec.dtype, np.dtype("int64"))

        # Test step.
        timestep = env.reset()
        self.assertTrue(timestep.first())
        timestep = env.step(1)
        self.assertEqual(timestep.reward, 1.0)
        self.assertTrue(np.isscalar(timestep.reward))
        self.assertEqual(timestep.observation.shape, (4,))
        env.close()

    def test_early_truncation(self):
        # Pendulum has no early termination condition. Recent versions of gym force
        # to use v1. We try both in case an earlier version is installed.
        try:
            gym_env = gym.make("Pendulum-v1")
        except:  # pylint: disable=bare-except
            gym_env = gym.make("Pendulum-v0")
        env = gym_wrapper.GymWrapper(gym_env)
        ts = env.reset()
        while not ts.last():
            ts = env.step(env.action_spec().generate_value())
        self.assertEqual(ts.discount, 1.0)
        self.assertTrue(np.isscalar(ts.reward))
        env.close()

    def test_multi_discrete(self):
        space = gym.spaces.MultiDiscrete([2, 3])
        spec = gym_wrapper._convert_to_spec(space)

        spec.validate([0, 0])
        spec.validate([1, 2])

        self.assertRaises(ValueError, spec.validate, [2, 2])
        self.assertRaises(ValueError, spec.validate, [1, 3])


@unittest.skipIf(SKIP_ATARI_TESTS, SKIP_ATARI_MESSAGE)
class AtariGymWrapperTest(absltest.TestCase):
    def test_pong(self):
        env = gym.make("PongNoFrameskip-v4", full_action_space=True)
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
        self.assertEqual(action_spec.dtype, np.dtype("int64"))

        # Test step.
        timestep = env.reset()
        self.assertTrue(timestep.first())
        _ = env.step([np.array(0)])
        env.close()


if __name__ == "__main__":
    absltest.main()
