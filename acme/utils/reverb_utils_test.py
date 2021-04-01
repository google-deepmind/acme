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

"""Tests for acme.utils.reverb_utils."""

from absl.testing import absltest
from acme.utils import reverb_utils
import reverb


class ReverbUtilsTest(absltest.TestCase):

  def test_make_replay_table_preserves_table_info(self):
    limiter = reverb.rate_limiters.SampleToInsertRatio(
        samples_per_insert=1, min_size_to_sample=2, error_buffer=(0, 10))
    table = reverb.Table(
        name='test',
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=10,
        rate_limiter=limiter)
    table_from_info = reverb_utils.make_replay_table_from_info(table.info)
    self.assertEqual(table_from_info.info, table.info)


if __name__ == '__main__':
  absltest.main()
