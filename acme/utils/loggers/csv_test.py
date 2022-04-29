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

"""Tests for csv logging."""

import csv
import os

from acme.testing import test_utils
from acme.utils import paths
from acme.utils.loggers import csv as csv_logger

from absl.testing import absltest
from absl.testing import parameterized

_TEST_INPUTS = [{
    'c': 'foo',
    'a': '1337',
    'b': '42.0001',
}, {
    'c': 'foo2',
    'a': '1338',
    'b': '43.0001',
}]


class CSVLoggingTest(test_utils.TestCase):

  def test_logging_input_is_directory(self):

    # Set up logger.
    directory = self.get_tempdir()
    label = 'test'
    logger = csv_logger.CSVLogger(directory_or_file=directory, label=label)

    # Write data and close.
    for inp in _TEST_INPUTS:
      logger.write(inp)
    logger.close()

    # Read back data.
    outputs = []
    with open(logger.file_path) as f:
      csv_reader = csv.DictReader(f)
      for row in csv_reader:
        outputs.append(dict(row))
    self.assertEqual(outputs, _TEST_INPUTS)

  @parameterized.parameters(True, False)
  def test_logging_input_is_file(self, add_uid: bool):

    # Set up logger.
    directory = paths.process_path(
        self.get_tempdir(), 'logs', 'my_label', add_uid=add_uid)
    file = open(os.path.join(directory, 'logs.csv'), 'a')
    logger = csv_logger.CSVLogger(directory_or_file=file, add_uid=add_uid)

    # Write data and close.
    for inp in _TEST_INPUTS:
      logger.write(inp)
    logger.close()

    # Logger doesn't close the file; caller must do this manually.
    self.assertFalse(file.closed)
    file.close()

    # Read back data.
    outputs = []
    with open(logger.file_path) as f:
      csv_reader = csv.DictReader(f)
      for row in csv_reader:
        outputs.append(dict(row))
    self.assertEqual(outputs, _TEST_INPUTS)

  def test_flush(self):

    logger = csv_logger.CSVLogger(self.get_tempdir(), flush_every=1)
    for inp in _TEST_INPUTS:
      logger.write(inp)

    # Read back data.
    outputs = []
    with open(logger.file_path) as f:
      csv_reader = csv.DictReader(f)
      for row in csv_reader:
        outputs.append(dict(row))
    self.assertEqual(outputs, _TEST_INPUTS)


if __name__ == '__main__':
  absltest.main()
