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

"""Tests for acme.datasets.numpy_iterator."""

import collections

from acme.datasets import numpy_iterator
import tensorflow as tf

from absl.testing import absltest


class NumpyIteratorTest(absltest.TestCase):

  def testBasic(self):
    ds = tf.data.Dataset.range(3)
    self.assertEqual([0, 1, 2], list(numpy_iterator.NumpyIterator(ds)))

  def testNestedStructure(self):
    point = collections.namedtuple('Point', ['x', 'y'])
    ds = tf.data.Dataset.from_tensor_slices({
        'a': ([1, 2], [3, 4]),
        'b': [5, 6],
        'c': point([7, 8], [9, 10])
    })
    self.assertEqual([{
        'a': (1, 3),
        'b': 5,
        'c': point(7, 9)
    }, {
        'a': (2, 4),
        'b': 6,
        'c': point(8, 10)
    }], list(numpy_iterator.NumpyIterator(ds)))

if __name__ == '__main__':
  absltest.main()
