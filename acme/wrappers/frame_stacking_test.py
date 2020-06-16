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
import tree


class FakeNonZeroObservationEnvironment(fakes.ContinuousEnvironment):
  """Fake environment with non-zero observations."""

  def _generate_fake_observation(self):
    original_observation = super()._generate_fake_observation()
    return tree.map_structure(np.ones_like, original_observation)


class FrameStackingTest(absltest.TestCase):

  def test_specs(self):
    original_env = FakeNonZeroObservationEnvironment()
    env = wrappers.FrameStackingWrapper(original_env, 2)

    original_observation_spec = original_env.observation_spec()
    expected_shape = original_observation_spec.shape + (2,)
    observation_spec = env.observation_spec()
    self.assertEqual(expected_shape, observation_spec.shape)

    expected_action_spec = original_env.action_spec()
    action_spec = env.action_spec()
    self.assertEqual(expected_action_spec, action_spec)

    expected_reward_spec = original_env.reward_spec()
    reward_spec = env.reward_spec()
    self.assertEqual(expected_reward_spec, reward_spec)

    expected_discount_spec = original_env.discount_spec()
    discount_spec = env.discount_spec()
    self.assertEqual(expected_discount_spec, discount_spec)

  def test_step(self):
    original_env = FakeNonZeroObservationEnvironment()
    env = wrappers.FrameStackingWrapper(original_env, 2)
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()

    timestep = env.reset()
    self.assertEqual(observation_spec.shape, timestep.observation.shape)
    self.assertTrue(np.all(timestep.observation[..., 0] == 0))

    timestep = env.step(action_spec.generate_value())
    self.assertEqual(observation_spec.shape, timestep.observation.shape)

  def test_second_reset(self):
    original_env = FakeNonZeroObservationEnvironment()
    env = wrappers.FrameStackingWrapper(original_env, 2)
    action_spec = env.action_spec()

    env.reset()
    env.step(action_spec.generate_value())
    timestep = env.reset()
    self.assertTrue(np.all(timestep.observation[..., 0] == 0))


if __name__ == '__main__':
  absltest.main()
