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

"""Tests for the delayed reward wrapper."""

from typing import Any
from acme import wrappers
from acme.testing import fakes
from dm_env import specs
import numpy as np
import tree

from absl.testing import absltest
from absl.testing import parameterized


def _episode_reward(env):
  timestep = env.reset()
  action_spec = env.action_spec()
  rng = np.random.RandomState(seed=1)
  reward = []
  while not timestep.last():
    timestep = env.step(rng.randint(action_spec.num_values))
    reward.append(timestep.reward)
  return reward


def _compare_nested_sequences(seq1, seq2):
  """Compare two sequences of arrays."""
  return all([(l == m).all() for l, m in zip(seq1, seq2)])


class _DiscreteEnvironmentOneReward(fakes.DiscreteEnvironment):
  """A fake discrete environement with constant reward of 1."""

  def _generate_fake_reward(self) -> Any:
    return tree.map_structure(lambda s: s.generate_value() + 1.,
                              self._spec.rewards)


class DelayedRewardTest(parameterized.TestCase):

  def test_noop(self):
    """Ensure when accumulation_period=1 it does not change anything."""
    base_env = _DiscreteEnvironmentOneReward(
        action_dtype=np.int64,
        reward_spec=specs.Array(dtype=np.float32, shape=()))
    wrapped_env = wrappers.DelayedRewardWrapper(base_env, accumulation_period=1)
    base_episode_reward = _episode_reward(base_env)
    wrapped_episode_reward = _episode_reward(wrapped_env)
    self.assertEqual(base_episode_reward, wrapped_episode_reward)

  def test_noop_composite_reward(self):
    """No-op test with composite rewards."""
    base_env = _DiscreteEnvironmentOneReward(
        action_dtype=np.int64,
        reward_spec=specs.Array(dtype=np.float32, shape=(2, 1)))
    wrapped_env = wrappers.DelayedRewardWrapper(base_env, accumulation_period=1)
    base_episode_reward = _episode_reward(base_env)
    wrapped_episode_reward = _episode_reward(wrapped_env)
    self.assertTrue(
        _compare_nested_sequences(base_episode_reward, wrapped_episode_reward))

  @parameterized.parameters(10, None)
  def test_same_episode_composite_reward(self, accumulation_period):
    """Ensure that wrapper does not change total reward."""
    base_env = _DiscreteEnvironmentOneReward(
        action_dtype=np.int64,
        reward_spec=specs.Array(dtype=np.float32, shape=()))
    wrapped_env = wrappers.DelayedRewardWrapper(
        base_env, accumulation_period=accumulation_period)
    base_episode_reward = _episode_reward(base_env)
    wrapped_episode_reward = _episode_reward(wrapped_env)
    self.assertTrue(
        (sum(base_episode_reward) == sum(wrapped_episode_reward)).all())


if __name__ == '__main__':
  absltest.main()
