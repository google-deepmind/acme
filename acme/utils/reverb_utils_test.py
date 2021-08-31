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
from acme import types
from acme.adders import reverb as reverb_adders
from acme.utils import reverb_utils
import numpy as np
import reverb
import tree


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
    new_table = reverb_utils.make_replay_table_from_info(table.info)
    new_info = new_table.info

    # table_worker_time is not set by the above utility since this is meant to
    # be monitoring information about any given table. So instead we copy this
    # so that the assertion below checks that everything else matches.

    # TODO(b/198297886): eliminate hasattr which exists for backwards compat.
    if hasattr(table.info, 'table_worker_time'):
      new_info.table_worker_time.sleeping_ms = (
          table.info.table_worker_time.sleeping_ms)

    self.assertEqual(new_info, table.info)

  _EMPTY_INFO = reverb.SampleInfo((), (), (), ())
  _DUMMY_OBS = np.array([[[0], [1], [2]]])
  _DUMMY_ACTION = np.array([[[3], [4], [5]]])
  _DUMMY_REWARD = np.array([[6, 7, 8]])
  _DUMMY_DISCOUNT = np.array([[.99, .99, .99]])
  _DUMMY_NEXT_OBS = np.array([[[1], [2], [0]]])

  def _create_dummy_steps(self):
    return reverb_adders.Step(
        observation=self._DUMMY_OBS,
        action=self._DUMMY_ACTION,
        reward=self._DUMMY_REWARD,
        discount=self._DUMMY_DISCOUNT,
        start_of_episode=True,
        extras=())

  def _create_dummy_transitions(self):
    return types.Transition(
        observation=self._DUMMY_OBS,
        action=self._DUMMY_ACTION,
        reward=self._DUMMY_REWARD,
        discount=self._DUMMY_DISCOUNT,
        next_observation=self._DUMMY_NEXT_OBS)

  def test_replay_sample_to_sars_transition_is_sequence(self):
    fake_sample = reverb.ReplaySample(
        info=self._EMPTY_INFO, data=self._create_dummy_steps())
    fake_transition = self._create_dummy_transitions()
    transition_from_sample = reverb_utils.replay_sample_to_sars_transition(
        fake_sample, is_sequence=True)
    tree.map_structure(np.testing.assert_array_equal, transition_from_sample,
                       fake_transition)


if __name__ == '__main__':
  absltest.main()
