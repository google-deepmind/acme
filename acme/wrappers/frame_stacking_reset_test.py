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

"""Frame histories must restart on implicit dm_env episode resets."""

from absl.testing import absltest
from absl.testing import parameterized
from acme.wrappers import frame_stacking
import dm_env
from dm_env import specs
import numpy as np
import tree


class CountingEpisodes(dm_env.Environment):
  """A deterministic two-step environment with standard automatic resets."""

  def __init__(self, nested=False, dtype=np.float32):
    self.episode = 0
    self.position = 0
    self.needs_reset = True
    self.nested = nested
    self.dtype = dtype

  def observation(self):
    value = 10 * self.episode + self.position
    vector = np.array([value, value + 1], dtype=self.dtype)
    return (
        {"vector": vector, "image": (np.full((2, 2, 1), value, self.dtype),)}
        if self.nested
        else vector
    )

  def reset(self):
    self.episode += 1
    self.position = 0
    self.needs_reset = False
    return dm_env.restart(self.observation())

  def step(self, action):
    del action
    if self.needs_reset:
      return self.reset()
    self.position += 1
    if self.position == 2:
      self.needs_reset = True
      return dm_env.termination(2.0, self.observation())
    return dm_env.transition(1.0, self.observation(), discount=0.5)

  def observation_spec(self):
    return tree.map_structure(
        lambda value: specs.Array(value.shape, value.dtype), self.observation()
    )

  def action_spec(self):
    return specs.DiscreteArray(1)


def expected_stack(observation, count, flatten):
  def stack(value):
    result = np.stack([np.zeros_like(value)] * (count - 1) + [value], axis=-1)
    return result.reshape(result.shape[:-2] + (-1,)) if flatten else result

  return tree.map_structure(stack, observation)


class FrameStackingResetTest(parameterized.TestCase):

  @parameterized.product(
      count=[1, 3], flatten=[False, True], nested=[False, True]
  )
  def test_automatic_reset_clears_every_previous_episode_frame(
      self, count, flatten, nested
  ):
    env = CountingEpisodes(nested=nested)
    wrapper = frame_stacking.FrameStackingWrapper(
        env, num_frames=count, flatten=flatten
    )
    first = wrapper.step(0)
    self.assertTrue(first.first())
    for _ in range(3):
      self.assertTrue(wrapper.step(0).mid())
      last = wrapper.step(0)
      self.assertTrue(last.last())
      self.assertEqual(last.reward, 2.0)
      self.assertEqual(last.discount, 0.0)
      reset = wrapper.step(0)
      self.assertTrue(reset.first())
      self.assertIsNone(reset.reward)
      self.assertIsNone(reset.discount)
      expected = expected_stack(env.observation(), count, flatten)
      tree.map_structure(
          np.testing.assert_array_equal, reset.observation, expected
      )
      tree.map_structure(
          lambda spec, value: spec.validate(value),
          wrapper.observation_spec(),
          reset.observation,
      )

  @parameterized.parameters(np.float32, np.uint8)
  def test_explicit_and_implicit_resets_agree(self, dtype):
    first_env = CountingEpisodes(dtype=dtype)
    second_env = CountingEpisodes(dtype=dtype)
    implicit = frame_stacking.FrameStackingWrapper(first_env, num_frames=4)
    explicit = frame_stacking.FrameStackingWrapper(second_env, num_frames=4)
    implicit.reset()
    explicit.reset()
    for _ in range(2):
      implicit.step(0)
      explicit.step(0)
    tree.map_structure(
        np.testing.assert_array_equal,
        implicit.step(0).observation,
        explicit.reset().observation,
    )

  def test_continuing_episode_keeps_real_history(self):
    wrapper = frame_stacking.FrameStackingWrapper(
        CountingEpisodes(), num_frames=3
    )
    wrapper.reset()
    mid = wrapper.step(0)
    np.testing.assert_array_equal(mid.observation, [[0, 10, 11], [0, 11, 12]])
    last = wrapper.step(0)
    np.testing.assert_array_equal(
        last.observation, [[10, 11, 12], [11, 12, 13]]
    )

  def test_manual_reset_in_the_middle_of_an_episode_discards_history(self):
    env = CountingEpisodes(nested=True)
    wrapper = frame_stacking.FrameStackingWrapper(env, num_frames=3)
    wrapper.reset()
    wrapper.step(0)
    result = wrapper.reset()
    tree.map_structure(
        np.testing.assert_array_equal,
        result.observation,
        expected_stack(env.observation(), 3, False),
    )


if __name__ == "__main__":
  absltest.main()
