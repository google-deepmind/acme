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

"""Tests for csv logging."""

import csv
import os

from absl.testing import absltest
from absl.testing import parameterized
from acme.testing import test_utils
from acme.utils import paths
from acme.utils.loggers import csv as csv_logger


class CSVLoggingTest(test_utils.TestCase):

  def test_logging_input_is_directory(self):
    inputs = [{
        'c': 'foo',
        'a': '1337',
        'b': '42.0001',
    }, {
        'c': 'foo2',
        'a': '1338',
        'b': '43.0001',
    }]
    directory = self.get_tempdir()
    label = 'test'
    logger = csv_logger.CSVLogger(directory_or_file=directory, label=label)
    for inp in inputs:
      logger.write(inp)
    outputs = []
    file_path = logger.file_path
    logger.close()

    with open(file_path) as f:
      csv_reader = csv.DictReader(f)
      for row in csv_reader:
        outputs.append(dict(row))
    self.assertEqual(outputs, inputs)

  @parameterized.parameters(True, False)
  def test_logging_input_is_file(self, add_uid):
    inputs = [{
        'c': 'foo',
        'a': '1337',
        'b': '42.0001',
    }, {
        'c': 'foo2',
        'a': '1338',
        'b': '43.0001',
    }]
    directory = paths.process_path(
        self.get_tempdir(), 'logs', 'my_label', add_uid=add_uid)
    file = open(os.path.join(directory, 'logs.csv'), 'a')
    logger = csv_logger.CSVLogger(directory_or_file=file, add_uid=add_uid)
    for inp in inputs:
      logger.write(inp)
    outputs = []
    file_path = logger.file_path
    logger.close()
    file.close()

    with open(file_path) as f:
      csv_reader = csv.DictReader(f)
      for row in csv_reader:
        outputs.append(dict(row))
    self.assertEqual(outputs, inputs)


if __name__ == '__main__':
  absltest.main()
