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

"""Tests for iterator_utils."""

from acme.utils import iterator_utils
import numpy as np

from absl.testing import absltest


class IteratorUtilsTest(absltest.TestCase):

  def test_iterator_zipping(self):

    def get_iters():
      x = iter(range(0, 10))
      y = iter(range(20, 30))
      return [x, y]

    zipped = zip(*get_iters())
    unzipped = iterator_utils.unzip_iterators(zipped, num_sub_iterators=2)
    expected_x, expected_y = get_iters()
    np.testing.assert_equal(list(unzipped[0]), list(expected_x))
    np.testing.assert_equal(list(unzipped[1]), list(expected_y))


if __name__ == '__main__':
  absltest.main()
