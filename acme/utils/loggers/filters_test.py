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

import time

from acme.utils.loggers import base
from acme.utils.loggers import filters

from absl.testing import absltest


# TODO(jaslanides): extract this to test_utils, or similar, for re-use.
class FakeLogger(base.Logger):
  """A fake logger for testing."""

  def __init__(self):
    self.data = []

  def write(self, data):
    self.data.append(data)

  @property
  def last_write(self):
    return self.data[-1]

  def close(self):
    pass


class GatedFilterTest(absltest.TestCase):

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


class TimeFilterTest(absltest.TestCase):

  def test_delta(self):
    logger = FakeLogger()
    filtered = filters.TimeFilter(logger, time_delta=0.1)

    # Logged.
    filtered.write({'foo': 1})
    self.assertIn('foo', logger.last_write)

    # *Not* logged.
    filtered.write({'bar': 2})
    self.assertNotIn('bar', logger.last_write)

    # Wait out delta.
    time.sleep(0.11)

    # Logged.
    filtered.write({'baz': 3})
    self.assertIn('baz', logger.last_write)

    self.assertLen(logger.data, 2)


class KeyFilterTest(absltest.TestCase):

  def test_keep_filter(self):
    logger = FakeLogger()
    filtered = filters.KeyFilter(logger, keep=('foo',))
    filtered.write({'foo': 'bar', 'baz': 12})
    row, *_ = logger.data
    self.assertIn('foo', row)
    self.assertNotIn('baz', row)

  def test_drop_filter(self):
    logger = FakeLogger()
    filtered = filters.KeyFilter(logger, drop=('foo',))
    filtered.write({'foo': 'bar', 'baz': 12})
    row, *_ = logger.data
    self.assertIn('baz', row)
    self.assertNotIn('foo', row)

  def test_bad_arguments(self):
    with self.assertRaises(ValueError):
      filters.KeyFilter(FakeLogger())
    with self.assertRaises(ValueError):
      filters.KeyFilter(FakeLogger(), keep=('a',), drop=('b',))


if __name__ == '__main__':
  absltest.main()
