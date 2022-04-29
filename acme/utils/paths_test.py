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

"""Tests for paths."""

from unittest import mock

from acme.testing import test_utils
import acme.utils.paths as paths

from absl.testing import flagsaver
from absl.testing import absltest


class PathTest(test_utils.TestCase):

  def test_process_path(self):
    root_directory = self.get_tempdir()
    with mock.patch.object(paths, 'get_unique_id') as mock_unique_id:
      mock_unique_id.return_value = ('test',)
      path = paths.process_path(root_directory, 'foo', 'bar')
    self.assertEqual(path, f'{root_directory}/test/foo/bar')

  def test_unique_id_with_flag(self):
    with flagsaver.flagsaver((paths.ACME_ID, 'test_flag')):
      self.assertEqual(paths.get_unique_id(), ('test_flag',))


if __name__ == '__main__':
  absltest.main()
