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

"""Tests for logging filters."""

from absl.testing import absltest
from acme.utils.loggers import base
from acme.utils.loggers import filters


# TODO(jaslanides): extract this to test_utils, or similar, for re-use.
class FakeLogger(base.Logger):
  """A fake logger for testing."""

  def __init__(self):
    self.data = []

  def write(self, data):
    self.data.append(data)

  def close(self):
    pass


class FiltersTest(absltest.TestCase):

  def test_logarithmic_filter(self):
    logger = FakeLogger()
    filtered = filters.GatedFilter.logarithmic(logger, n=10)
    for t in range(100):
      filtered.write({'t': t})
    rows = [row['t'] for row in logger.data]
    self.assertEqual(rows, [*range(10), *range(10, 100, 10)])

  def test_periodic_filter(self):
    logger = FakeLogger()
    filtered = filters.GatedFilter.periodic(logger, interval=10)
    for t in range(100):
      filtered.write({'t': t})
    rows = [row['t'] for row in logger.data]
    self.assertEqual(rows, list(range(0, 100, 10)))


if __name__ == '__main__':
  absltest.main()
