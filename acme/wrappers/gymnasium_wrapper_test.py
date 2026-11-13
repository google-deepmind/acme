# Copyright 2024 DeepMind Technologies Limited. All rights reserved.
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

"""Tests for gymnasium_wrapper."""

import unittest
from unittest import mock

from absl.testing import absltest
from acme import specs
import dm_env
import numpy as np

SKIP_GYMNASIUM_TESTS = False
SKIP_GYMNASIUM_MESSAGE = 'gymnasium not installed.'

try:
  import gymnasium
  from gymnasium import spaces
  from acme.wrappers import gymnasium_wrapper
except (ImportError, ModuleNotFoundError):
  SKIP_GYMNASIUM_TESTS = True


class FakeGymnasiumEnv:
  """Mock Gymnasium environment for testing."""

  def __init__(self):
    self.observation_space = spaces.Box(
        low=-1.0, high=1.0, shape=(4,), dtype=np.float32
    )
    self.action_space = spaces.Discrete(2)
    self._step_count = 0

  def reset(self, seed=None, options=None):
    self._step_count = 0
    return np.zeros((4,), dtype=np.float32), {'reset_seed': seed}

  def step(self, action):
    self._step_count += 1
    obs = np.ones((4,), dtype=np.float32) * self._step_count
    reward = 1.0
    terminated = self._step_count >= 3
    truncated = False
    info = {'step_count': self._step_count}
    return obs, reward, terminated, truncated, info

  def close(self):
    pass


@unittest.skipIf(SKIP_GYMNASIUM_TESTS, SKIP_GYMNASIUM_MESSAGE)
class GymnasiumWrapperTest(absltest.TestCase):

  def test_specs(self):
    env = gymnasium_wrapper.GymnasiumWrapper(FakeGymnasiumEnv())
    obs_spec = env.observation_spec()
    act_spec = env.action_spec()

    self.assertIsInstance(obs_spec, specs.BoundedArray)
    self.assertEqual(obs_spec.shape, (4,))
    self.assertEqual(obs_spec.dtype, np.float32)

    self.assertIsInstance(act_spec, specs.DiscreteArray)
    self.assertEqual(act_spec.num_values, 2)

  def test_reset(self):
    env = gymnasium_wrapper.GymnasiumWrapper(FakeGymnasiumEnv())
    timestep = env.reset(seed=42)

    self.assertTrue(timestep.first())
    np.testing.assert_array_equal(timestep.observation, np.zeros((4,)))
    self.assertEqual(env.get_info(), {'reset_seed': 42})

  def test_step_and_termination(self):
    env = gymnasium_wrapper.GymnasiumWrapper(FakeGymnasiumEnv())
    _ = env.reset()

    # Step 1: transition
    ts1 = env.step(0)
    self.assertTrue(ts1.mid())
    self.assertEqual(ts1.reward, 1.0)
    self.assertEqual(ts1.discount, 1.0)

    # Step 2: transition
    ts2 = env.step(1)
    self.assertTrue(ts2.mid())

    # Step 3: termination
    ts3 = env.step(0)
    self.assertTrue(ts3.last())
    self.assertEqual(ts3.discount, 0.0)

    # Step 4: auto-reset after termination
    ts4 = env.step(0)
    self.assertTrue(ts4.first())

  def test_truncation(self):
    fake_env = FakeGymnasiumEnv()

    def truncated_step(action):
      return np.zeros((4,), dtype=np.float32), 0.5, False, True, {'time_limit': True}

    fake_env.step = truncated_step
    env = gymnasium_wrapper.GymnasiumWrapper(fake_env)
    _ = env.reset()
    ts = env.step(0)

    self.assertTrue(ts.last())
    self.assertEqual(ts.step_type, dm_env.StepType.LAST)
    self.assertEqual(ts.discount, 1.0)  # Truncation preserves discount

  def test_space_conversions(self):
    # MultiBinary
    mb_space = spaces.MultiBinary(5)
    spec = gymnasium_wrapper._convert_to_spec(mb_space)
    self.assertIsInstance(spec, specs.BoundedArray)
    self.assertEqual(spec.shape, (5,))

    # MultiDiscrete
    md_space = spaces.MultiDiscrete([3, 4])
    spec = gymnasium_wrapper._convert_to_spec(md_space)
    self.assertIsInstance(spec, specs.BoundedArray)
    self.assertEqual(spec.shape, (2,))

    # Dict
    d_space = spaces.Dict({'a': spaces.Discrete(3), 'b': spaces.Box(0, 1, shape=(2,))})
    spec = gymnasium_wrapper._convert_to_spec(d_space)
    self.assertIsInstance(spec, dict)
    self.assertIn('a', spec)
    self.assertIn('b', spec)

    # Tuple
    t_space = spaces.Tuple((spaces.Discrete(2), spaces.Box(0, 1, shape=(3,))))
    spec = gymnasium_wrapper._convert_to_spec(t_space)
    self.assertIsInstance(spec, tuple)
    self.assertEqual(len(spec), 2)


if __name__ == '__main__':
  absltest.main()
