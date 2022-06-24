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

"""Tests for dataset."""

from acme.agents.jax.mbop import dataset as dataset_lib
import rlds
from rlds.transformations import transformations_testlib
import tensorflow as tf

from absl.testing import absltest


def sample_episode() -> rlds.Episode:
  """Returns a sample episode."""
  steps = {
      rlds.OBSERVATION: [
          [1, 1],
          [2, 2],
          [3, 3],
          [4, 4],
          [5, 5],
      ],
      rlds.ACTION: [[1], [2], [3], [4], [5]],
      rlds.REWARD: [1.0, 2.0, 3.0, 4.0, 5.0],
      rlds.DISCOUNT: [1, 1, 1, 1, 1],
      rlds.IS_FIRST: [True, False, False, False, False],
      rlds.IS_LAST: [False, False, False, False, True],
      rlds.IS_TERMINAL: [False, False, False, False, True],
  }
  return {rlds.STEPS: tf.data.Dataset.from_tensor_slices(steps)}


class DatasetTest(transformations_testlib.TransformationsTest):

  def test_episode_to_timestep_batch(self):
    batched = dataset_lib.episode_to_timestep_batch(
        sample_episode(), return_horizon=2)

    # Scalars should be expanded and the n-step return should be present. Each
    # element of a step should be a triplet containing the previous, current and
    # next values of the corresponding fields. Since the return horizon is 2 and
    # the number of steps in the episode is 5, there can be only 2 triplets for
    # time steps 1 and 2.
    expected_steps = {
        rlds.OBSERVATION: [
            [[1, 1], [2, 2], [3, 3]],
            [[2, 2], [3, 3], [4, 4]],
        ],
        rlds.ACTION: [
            [[1], [2], [3]],
            [[2], [3], [4]],
        ],
        rlds.REWARD: [
            [[1.0], [2.0], [3.0]],
            [[2.0], [3.0], [4.0]],
        ],
        rlds.DISCOUNT: [
            [[1], [1], [1]],
            [[1], [1], [1]],
        ],
        rlds.IS_FIRST: [
            [[True], [False], [False]],
            [[False], [False], [False]],
        ],
        rlds.IS_LAST: [
            [[False], [False], [False]],
            [[False], [False], [False]],
        ],
        rlds.IS_TERMINAL: [
            [[False], [False], [False]],
            [[False], [False], [False]],
        ],
        dataset_lib.N_STEP_RETURN: [
            [[3.0], [5.0], [7.0]],
            [[5.0], [7.0], [9.0]],
        ],
    }

    self.expect_equal_datasets(
        batched, tf.data.Dataset.from_tensor_slices(expected_steps))

  def test_episode_to_timestep_batch_episode_return(self):
    batched = dataset_lib.episode_to_timestep_batch(
        sample_episode(), return_horizon=3, calculate_episode_return=True)

    expected_steps = {
        rlds.OBSERVATION: [[[1, 1], [2, 2], [3, 3]]],
        rlds.ACTION: [[[1], [2], [3]]],
        rlds.REWARD: [[[1.0], [2.0], [3.0]]],
        rlds.DISCOUNT: [[[1], [1], [1]]],
        rlds.IS_FIRST: [[[True], [False], [False]]],
        rlds.IS_LAST: [[[False], [False], [False]]],
        rlds.IS_TERMINAL: [[[False], [False], [False]]],
        dataset_lib.N_STEP_RETURN: [[[6.0], [9.0], [12.0]]],
        # This should match to the sum of the rewards in the input.
        dataset_lib.EPISODE_RETURN: [[[15.0], [15.0], [15.0]]],
    }

    self.expect_equal_datasets(
        batched, tf.data.Dataset.from_tensor_slices(expected_steps))

  def test_episode_to_timestep_batch_no_return_horizon(self):
    batched = dataset_lib.episode_to_timestep_batch(
        sample_episode(), return_horizon=1)

    expected_steps = {
        rlds.OBSERVATION: [
            [[1, 1], [2, 2], [3, 3]],
            [[2, 2], [3, 3], [4, 4]],
            [[3, 3], [4, 4], [5, 5]],
        ],
        rlds.ACTION: [
            [[1], [2], [3]],
            [[2], [3], [4]],
            [[3], [4], [5]],
        ],
        rlds.REWARD: [
            [[1.0], [2.0], [3.0]],
            [[2.0], [3.0], [4.0]],
            [[3.0], [4.0], [5.0]],
        ],
        rlds.DISCOUNT: [
            [[1], [1], [1]],
            [[1], [1], [1]],
            [[1], [1], [1]],
        ],
        rlds.IS_FIRST: [
            [[True], [False], [False]],
            [[False], [False], [False]],
            [[False], [False], [False]],
        ],
        rlds.IS_LAST: [
            [[False], [False], [False]],
            [[False], [False], [False]],
            [[False], [False], [True]],
        ],
        rlds.IS_TERMINAL: [
            [[False], [False], [False]],
            [[False], [False], [False]],
            [[False], [False], [True]],
        ],
        # n-step return should be equal to the rewards.
        dataset_lib.N_STEP_RETURN: [
            [[1.0], [2.0], [3.0]],
            [[2.0], [3.0], [4.0]],
            [[3.0], [4.0], [5.0]],
        ],
    }

    self.expect_equal_datasets(
        batched, tf.data.Dataset.from_tensor_slices(expected_steps))

  def test_episode_to_timestep_batch_drop_return_horizon(self):
    steps = {
        rlds.OBSERVATION: [[1], [2], [3], [4], [5], [6]],
        rlds.REWARD: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }
    episode = {rlds.STEPS: tf.data.Dataset.from_tensor_slices(steps)}

    batched = dataset_lib.episode_to_timestep_batch(
        episode,
        return_horizon=2,
        calculate_episode_return=True,
        drop_return_horizon=True)

    # The two steps of the episode should be dropped. There will be 4 steps left
    # and since the return horizon is 2, only a single 3-batched step should be
    # emitted. The episode return should be the sum of the rewards of the first
    # 4 steps.
    expected_steps = {
        rlds.OBSERVATION: [[[1], [2], [3]]],
        rlds.REWARD: [[[1.0], [2.0], [3.0]]],
        dataset_lib.N_STEP_RETURN: [[[3.0], [5.0], [7.0]]],
        dataset_lib.EPISODE_RETURN: [[[10.0], [10.0], [10.0]]],
    }

    self.expect_equal_datasets(
        batched, tf.data.Dataset.from_tensor_slices(expected_steps))


if __name__ == '__main__':
  absltest.main()
