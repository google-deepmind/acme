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

"""Tests for sequence adders."""

from acme.adders.reverb import sequence as adders
from acme.adders.reverb import test_cases
from acme.adders.reverb import test_utils

from absl.testing import absltest
from absl.testing import parameterized


class SequenceAdderTest(test_utils.AdderTestMixin, parameterized.TestCase):

  @parameterized.named_parameters(*test_cases.TEST_CASES_FOR_SEQUENCE_ADDER)
  def test_adder(self,
                 sequence_length: int,
                 period: int,
                 first,
                 steps,
                 expected_sequences,
                 end_behavior: adders.EndBehavior = adders.EndBehavior.ZERO_PAD,
                 repeat_episode_times: int = 1):
    adder = adders.SequenceAdder(
        self.client,
        sequence_length=sequence_length,
        period=period,
        end_of_episode_behavior=end_behavior)
    super().run_test_adder(
        adder=adder,
        first=first,
        steps=steps,
        expected_items=expected_sequences,
        repeat_episode_times=repeat_episode_times,
        end_behavior=end_behavior,
        signature=adder.signature(*test_utils.get_specs(steps[0])))

  @parameterized.parameters(
      (True, True, adders.EndBehavior.ZERO_PAD),
      (False, True, adders.EndBehavior.TRUNCATE),
      (False, False, adders.EndBehavior.CONTINUE),
  )
  def test_end_of_episode_behavior_set_correctly(self, pad_end_of_episode,
                                                 break_end_of_episode,
                                                 expected_behavior):
    adder = adders.SequenceAdder(
        self.client,
        sequence_length=5,
        period=3,
        pad_end_of_episode=pad_end_of_episode,
        break_end_of_episode=break_end_of_episode)
    self.assertEqual(adder._end_of_episode_behavior, expected_behavior)


if __name__ == '__main__':
  absltest.main()
