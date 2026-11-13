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

"""No-op starts should return a true FIRST timestep with no prior reward."""

from types import SimpleNamespace

from absl.testing import absltest
from absl.testing import parameterized
from acme.wrappers import noop_starts
import dm_env
from dm_env import specs
import numpy as np


class ShortEpisodeEnvironment(dm_env.Environment):
  """Two no-op steps per episode, with a nonzero MID reward/discount."""

  def __init__(self):
    self.episode = 0
    self.step_count = 0
    self.steps = []

  def reset(self):
    self.episode += 1
    self.step_count = 0
    return dm_env.restart({'episode': self.episode, 'frame': 0})

  def step(self, action):
    self.steps.append(action)
    self.step_count += 1
    observation = {'episode': self.episode, 'frame': self.step_count}
    if self.step_count >= 2:
      return dm_env.termination(np.float32(4), observation)
    return dm_env.transition(np.float32(3), observation, discount=.75)

  def action_spec(self):
    return specs.DiscreteArray(2)

  def observation_spec(self):
    return {'episode': specs.Array((), np.int32),
            'frame': specs.Array((), np.int32)}

  def reward_spec(self):
    return specs.Array((), np.float32)

  def discount_spec(self):
    return specs.BoundedArray((), np.float32, 0., 1.)


class NoopStartsFirstTimestepTest(parameterized.TestCase):

  @parameterized.parameters(
      (0, 1, 0),
      (1, 1, 1),
      (2, 2, 0),
      (3, 2, 1),
  )
  def test_noop_reset_clears_reward_and_discount(
      self, noops, expected_episode, expected_frame
  ):
    base = ShortEpisodeEnvironment()
    wrapper = noop_starts.NoopStartsWrapper(
        base, noop_action=1, noop_max=3, seed=0
    )
    wrapper.np_random = SimpleNamespace(randint=lambda high: noops)

    first = wrapper.reset()

    self.assertTrue(first.first())
    self.assertIsNone(first.reward)
    self.assertIsNone(first.discount)
    self.assertEqual(first.observation, {
        'episode': expected_episode, 'frame': expected_frame
    })
    self.assertEqual(base.steps, [1] * noops)

  def test_next_step_returns_underlying_transition_unchanged(self):
    base = ShortEpisodeEnvironment()
    wrapper = noop_starts.NoopStartsWrapper(
        base, noop_action=0, noop_max=1, seed=1
    )
    wrapper.np_random = SimpleNamespace(randint=lambda high: 1)
    first = wrapper.reset()
    self.assertTrue(first.first())
    step = wrapper.step(1)
    self.assertTrue(step.last())
    self.assertEqual(step.reward, 4)
    self.assertEqual(step.discount, 0)
    self.assertEqual(base.steps, [0, 1])

  def test_reset_preserves_observation_after_early_termination(self):
    base = ShortEpisodeEnvironment()
    wrapper = noop_starts.NoopStartsWrapper(
        base, noop_action=0, noop_max=5, seed=1
    )
    wrapper.np_random = SimpleNamespace(randint=lambda high: 5)
    first = wrapper.reset()
    self.assertTrue(first.first())
    self.assertIsNone(first.reward)
    self.assertIsNone(first.discount)
    self.assertEqual(first.observation, {'episode': 3, 'frame': 1})


if __name__ == '__main__':
  absltest.main()
