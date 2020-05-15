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

"""Tests for the single precision wrapper."""

from absl.testing import absltest

from acme import wrappers
from acme.testing import fakes

import numpy as np


class SinglePrecisionTest(absltest.TestCase):

  def test_continuous(self):
    env = wrappers.SinglePrecisionWrapper(
        fakes.ContinuousEnvironment(
            action_dim=0, dtype=np.float64, reward_dtype=np.float64))

    self.assertTrue(np.issubdtype(env.observation_spec().dtype, np.float32))
    self.assertTrue(np.issubdtype(env.action_spec().dtype, np.float32))
    self.assertTrue(np.issubdtype(env.reward_spec().dtype, np.float32))
    self.assertTrue(np.issubdtype(env.discount_spec().dtype, np.float32))

    timestep = env.reset()
    self.assertEqual(timestep.reward, None)
    self.assertEqual(timestep.discount, None)
    self.assertTrue(np.issubdtype(timestep.observation.dtype, np.float32))

    timestep = env.step(0.0)
    self.assertTrue(np.issubdtype(timestep.reward.dtype, np.float32))
    self.assertTrue(np.issubdtype(timestep.discount.dtype, np.float32))
    self.assertTrue(np.issubdtype(timestep.observation.dtype, np.float32))

  def test_discrete(self):
    env = wrappers.SinglePrecisionWrapper(
        fakes.DiscreteEnvironment(
            action_dtype=np.int64, obs_dtype=np.int64, reward_dtype=np.float64))

    self.assertTrue(np.issubdtype(env.observation_spec().dtype, np.int32))
    self.assertTrue(np.issubdtype(env.action_spec().dtype, np.int32))
    self.assertTrue(np.issubdtype(env.reward_spec().dtype, np.float32))
    self.assertTrue(np.issubdtype(env.discount_spec().dtype, np.float32))

    timestep = env.reset()
    self.assertEqual(timestep.reward, None)
    self.assertEqual(timestep.discount, None)
    self.assertTrue(np.issubdtype(timestep.observation.dtype, np.int32))

    timestep = env.step(0)
    self.assertTrue(np.issubdtype(timestep.reward.dtype, np.float32))
    self.assertTrue(np.issubdtype(timestep.discount.dtype, np.float32))
    self.assertTrue(np.issubdtype(timestep.observation.dtype, np.int32))


if __name__ == '__main__':
  absltest.main()
