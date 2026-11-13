# Copyright 2026 DeepMind Technologies Limited. All rights reserved.
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

"""Delayed rewards across episodes that automatically reset on step()."""

from absl.testing import absltest
from absl.testing import parameterized
from acme.wrappers import delayed_reward
import dm_env
from dm_env import specs
import numpy as np
import tree


class AutoResetEnvironment(dm_env.Environment):
  """Two-step dm_env episodes, with a FIRST on step() after LAST."""

  def __init__(self, nested=False, truncate=False):
    self._nested = nested
    self._truncate = truncate
    self._episode = 0
    self._step = 0

  def reset(self):
    self._episode += 1
    self._step = 0
    return dm_env.restart(self._episode)

  def step(self, action):
    del action
    if self._step == 2 or self._episode == 0:
      return self.reset()
    self._step += 1
    value = np.float32(self._step + 10 * (self._episode - 1))
    reward = {'left': value, 'right': np.array([value, 2 * value])} if self._nested else value
    if self._step == 2:
      if self._truncate:
        return dm_env.truncation(reward, self._episode, discount=0.75)
      return dm_env.termination(reward, self._episode)
    return dm_env.transition(reward, self._episode)

  def reward_spec(self):
    if self._nested:
      return {'left': specs.Array((), np.float32),
              'right': specs.Array((2,), np.float32)}
    return specs.Array((), np.float32)

  def observation_spec(self):
    return specs.Array((), np.int32)

  def action_spec(self):
    return specs.DiscreteArray(1)

  def discount_spec(self):
    return specs.BoundedArray((), np.float32, 0., 1.)

  def close(self):
    pass


class DelayedRewardAutoResetTest(parameterized.TestCase):

  @parameterized.product(nested=(False, True), truncate=(False, True))
  def test_automatic_reset_clears_pending_rewards(self, nested, truncate):
    env = delayed_reward.DelayedRewardWrapper(
        AutoResetEnvironment(nested=nested, truncate=truncate),
        accumulation_period=4,
    )

    def assert_reward_equal(actual, expected):
      tree.map_structure(np.testing.assert_array_equal, actual, expected)

    first = env.reset()
    self.assertTrue(first.first())
    self.assertIsNone(first.reward)

    for episode in range(1, 4):
      intermediate = env.step(0)
      self.assertTrue(intermediate.mid())
      assert_reward_equal(intermediate.reward, tree.map_structure(
          lambda x: np.zeros_like(x), env.reward_spec()
      ))
      last = env.step(0)
      self.assertTrue(last.last())
      expected_sum = np.float32(3 + 20 * (episode - 1))
      if nested:
        expected = {'left': expected_sum,
                    'right': np.array([expected_sum, 2 * expected_sum])}
      else:
        expected = expected_sum
      assert_reward_equal(last.reward, expected)
      if truncate:
        self.assertEqual(last.discount, 0.75)

      new_first = env.step(0)
      self.assertTrue(new_first.first())
      self.assertIsNone(new_first.reward)
      self.assertEqual(new_first.observation, episode + 1)

  @parameterized.parameters(1, 2, None)
  def test_explicit_reset_still_starts_a_fresh_accumulation(self, period):
    env = delayed_reward.DelayedRewardWrapper(
        AutoResetEnvironment(), accumulation_period=period
    )
    env.reset()
    env.step(0)
    env.reset()
    first_reward = env.step(0).reward
    if period == 1:
      self.assertEqual(first_reward, 11)
    else:
      self.assertEqual(first_reward, 0)
    final_reward = env.step(0).reward
    self.assertEqual(final_reward, 12 if period == 1 else (23 if period in (2, None) else 0))


if __name__ == '__main__':
  absltest.main()
