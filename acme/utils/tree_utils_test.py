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

"""Tests for tree_utils."""

from absl.testing import absltest
from acme.utils import tree_utils
import numpy as np
import tree

TEST_SEQUENCE = [
    {
        'action': np.array([1.0]),
        'observation': (np.array([0.0, 1.0, 2.0]),),
        'reward': np.array(1.0),
    },
    {
        'action': np.array([0.5]),
        'observation': (np.array([1.0, 2.0, 3.0]),),
        'reward': np.array(0.0),
    },
    {
        'action': np.array([0.3]),
        'observation': (np.array([2.0, 3.0, 4.0]),),
        'reward': np.array(0.5),
    },
]


class SequenceStackTest(absltest.TestCase):
  """Tests for various tree utilities."""

  def test_stack_sequence_fields(self):
    """Tests that `stack_sequence_fields` behaves correctly on nested data."""

    stacked = tree_utils.stack_sequence_fields(TEST_SEQUENCE)

    # Check that the stacked output has the correct structure.
    tree.assert_same_structure(stacked, TEST_SEQUENCE[0])

    # Check that the leaves have the correct array shapes.
    self.assertEqual(stacked['action'].shape, (3, 1))
    self.assertEqual(stacked['observation'][0].shape, (3, 3))
    self.assertEqual(stacked['reward'].shape, (3,))

    # Check values.
    self.assertEqual(stacked['observation'][0].tolist(), [
        [0., 1., 2.],
        [1., 2., 3.],
        [2., 3., 4.],
    ])
    self.assertEqual(stacked['action'].tolist(), [[1.], [0.5], [0.3]])
    self.assertEqual(stacked['reward'].tolist(), [1., 0., 0.5])

  def test_unstack_sequence_fields(self):
    """Tests that `unstack_sequence_fields(stack_sequence_fields(x)) == x`."""
    stacked = tree_utils.stack_sequence_fields(TEST_SEQUENCE)
    batch_size = len(TEST_SEQUENCE)
    unstacked = tree_utils.unstack_sequence_fields(stacked, batch_size)
    tree.map_structure(np.testing.assert_array_equal, unstacked, TEST_SEQUENCE)


if __name__ == '__main__':
  absltest.main()
