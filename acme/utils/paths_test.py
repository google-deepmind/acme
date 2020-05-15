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

"""Tests for paths."""

from unittest import mock
import uuid

from absl.testing import absltest
from acme.testing import test_utils
import acme.utils.paths as paths


class PathTest(test_utils.TestCase):

  def test_process_path(self):
    root_directory = self.get_tempdir()
    with mock.patch.object(uuid, 'uuid1') as mock_uuid:
      mock_uuid.return_value = 123
      path = paths.process_path(root_directory, 'foo', 'bar')
      self.assertEqual(path, f'{root_directory}/123/foo/bar')


if __name__ == '__main__':
  absltest.main()



