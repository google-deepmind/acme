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

"""Tests for sequence adders."""

from absl.testing import absltest
from absl.testing import parameterized

from acme.adders.reverb import sequence as adders
from acme.adders.reverb import test_utils

import dm_env

TEST_CASES = [
    dict(
        testcase_name='PeriodOne',
        sequence_length=3,
        period=1,
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.transition(reward=3.0, observation=3)),
            (0, dm_env.transition(reward=5.0, observation=4)),
            (0, dm_env.termination(reward=7.0, observation=5)),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode, extra)
            [
                (1, 0, 2.0, 1.0, True, ()),
                (2, 0, 3.0, 1.0, False, ()),
                (3, 0, 5.0, 1.0, False, ()),
            ],
            [
                (2, 0, 3.0, 1.0, False, ()),
                (3, 0, 5.0, 1.0, False, ()),
                (4, 0, 7.0, 0.0, False, ()),
            ],
            [
                (3, 0, 5.0, 1.0, False, ()),
                (4, 0, 7.0, 0.0, False, ()),
                (5, 0, 0.0, 0.0, False, ()),
            ],
        ),
    ),
    dict(
        testcase_name='PeriodTwo',
        sequence_length=3,
        period=2,
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.transition(reward=3.0, observation=3)),
            (0, dm_env.transition(reward=5.0, observation=4)),
            (0, dm_env.termination(reward=7.0, observation=5)),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode, extra)
            [
                (1, 0, 2.0, 1.0, True, ()),
                (2, 0, 3.0, 1.0, False, ()),
                (3, 0, 5.0, 1.0, False, ()),
            ],
            [
                (3, 0, 5.0, 1.0, False, ()),
                (4, 0, 7.0, 0.0, False, ()),
                (5, 0, 0.0, 0.0, False, ()),
            ],
        ),
    ),
    dict(
        testcase_name='EarlyTerminationPeriodOne',
        sequence_length=3,
        period=1,
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.termination(reward=3.0, observation=3)),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode, extra)
            [
                (1, 0, 2.0, 1.0, True, ()),
                (2, 0, 3.0, 0.0, False, ()),
                (3, 0, 0.0, 0.0, False, ()),
            ],),
    ),
    dict(
        testcase_name='EarlyTerminationPeriodTwo',
        sequence_length=3,
        period=2,
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.termination(reward=3.0, observation=3)),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode, extra)
            [
                (1, 0, 2.0, 1.0, True, ()),
                (2, 0, 3.0, 0.0, False, ()),
                (3, 0, 0.0, 0.0, False, ()),
            ],),
    ),
    dict(
        testcase_name='EarlyTerminationPaddingPeriodOne',
        sequence_length=4,
        period=1,
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.termination(reward=3.0, observation=3)),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode, extra)
            [
                (1, 0, 2.0, 1.0, True, ()),
                (2, 0, 3.0, 0.0, False, ()),
                (3, 0, 0.0, 0.0, False, ()),
                (0, 0, 0.0, 0.0, False, ()),
            ],),
    ),
    dict(
        testcase_name='EarlyTerminationPaddingPeriodTwo',
        sequence_length=4,
        period=2,
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.termination(reward=3.0, observation=3)),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode, extra)
            [
                (1, 0, 2.0, 1.0, True, ()),
                (2, 0, 3.0, 0.0, False, ()),
                (3, 0, 0.0, 0.0, False, ()),
                (0, 0, 0.0, 0.0, False, ()),
            ],),
    ),
    dict(
        testcase_name='EarlyTerminationNoPadding',
        sequence_length=4,
        period=1,
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.termination(reward=3.0, observation=3)),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode, extra)
            [
                (1, 0, 2.0, 1.0, True, ()),
                (2, 0, 3.0, 0.0, False, ()),
                (3, 0, 0.0, 0.0, False, ()),
            ],),
        end_behavior=adders.EndBehavior.TRUNCATE,
    ),
    dict(
        testcase_name='LongEpisodePadding',
        sequence_length=3,
        period=3,
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.transition(reward=3.0, observation=3)),
            (0, dm_env.transition(reward=5.0, observation=4)),
            (0, dm_env.transition(reward=7.0, observation=5)),
            (0, dm_env.transition(reward=9.0, observation=6)),
            (0, dm_env.transition(reward=11.0, observation=7)),
            (0, dm_env.termination(reward=13.0, observation=8)),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode, extra)
            [
                (1, 0, 2.0, 1.0, True, ()),
                (2, 0, 3.0, 1.0, False, ()),
                (3, 0, 5.0, 1.0, False, ()),
            ],
            [
                (4, 0, 7.0, 1.0, False, ()),
                (5, 0, 9.0, 1.0, False, ()),
                (6, 0, 11.0, 1.0, False, ()),
            ],
            [
                (7, 0, 13.0, 0.0, False, ()),
                (8, 0, 0.0, 0.0, False, ()),
                (0, 0, 0.0, 0.0, False, ()),
            ],
        ),
    ),
    dict(
        testcase_name='LongEpisodeNoPadding',
        sequence_length=3,
        period=3,
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.transition(reward=3.0, observation=3)),
            (0, dm_env.transition(reward=5.0, observation=4)),
            (0, dm_env.transition(reward=7.0, observation=5)),
            (0, dm_env.transition(reward=9.0, observation=6)),
            (0, dm_env.transition(reward=11.0, observation=7)),
            (0, dm_env.termination(reward=13.0, observation=8)),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode, extra)
            [
                (1, 0, 2.0, 1.0, True, ()),
                (2, 0, 3.0, 1.0, False, ()),
                (3, 0, 5.0, 1.0, False, ()),
            ],
            [
                (4, 0, 7.0, 1.0, False, ()),
                (5, 0, 9.0, 1.0, False, ()),
                (6, 0, 11.0, 1.0, False, ()),
            ],
            [
                (7, 0, 13.0, 0.0, False, ()),
                (8, 0, 0.0, 0.0, False, ()),
            ],
        ),
        end_behavior=adders.EndBehavior.TRUNCATE,
    ),
    dict(
        testcase_name='NonBreakingSequenceOnEpisodeReset',
        sequence_length=3,
        period=2,
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.transition(reward=3.0, observation=3)),
            (0, dm_env.transition(reward=5.0, observation=4)),
            (0, dm_env.transition(reward=7.0, observation=5)),
            (0, dm_env.transition(reward=9.0, observation=6)),
            (0, dm_env.transition(reward=11.0, observation=7)),
            (0, dm_env.termination(reward=13.0, observation=8)),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode, extra)
            [
                (1, 0, 2.0, 1.0, True, ()),
                (2, 0, 3.0, 1.0, False, ()),
                (3, 0, 5.0, 1.0, False, ()),
            ],
            [
                (3, 0, 5.0, 1.0, False, ()),
                (4, 0, 7.0, 1.0, False, ()),
                (5, 0, 9.0, 1.0, False, ()),
            ],
            [
                (5, 0, 9.0, 1.0, False, ()),
                (6, 0, 11.0, 1.0, False, ()),
                (7, 0, 13.0, 0.0, False, ()),
            ],
        ),
        end_behavior=adders.EndBehavior.CONTINUE,
        repeat_episode_times=1
    ),
    dict(
        testcase_name='NonBreakingSequenceMultipleTerminatedEpisodes',
        sequence_length=3,
        period=2,
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.transition(reward=3.0, observation=3)),
            (0, dm_env.transition(reward=5.0, observation=4)),
            (0, dm_env.transition(reward=7.0, observation=5)),
            (0, dm_env.transition(reward=9.0, observation=6)),
            (0, dm_env.termination(reward=13.0, observation=7)),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode, extra)
            [
                (1, 0, 2.0, 1.0, True, ()),
                (2, 0, 3.0, 1.0, False, ()),
                (3, 0, 5.0, 1.0, False, ()),
            ],
            [
                (3, 0, 5.0, 1.0, False, ()),
                (4, 0, 7.0, 1.0, False, ()),
                (5, 0, 9.0, 1.0, False, ()),
            ],
            [
                (5, 0, 9.0, 1.0, False, ()),
                (6, 0, 13.0, 0.0, False, ()),
                (7, 0, 0.0, 0.0, False, ()),
            ],
            [
                (7, 0, 0.0, 0.0, False, ()),
                (1, 0, 2.0, 1.0, True, ()),
                (2, 0, 3.0, 1.0, False, ()),
            ],
            [
                (2, 0, 3.0, 1.0, False, ()),
                (3, 0, 5.0, 1.0, False, ()),
                (4, 0, 7.0, 1.0, False, ()),
            ],
            [
                (4, 0, 7.0, 1.0, False, ()),
                (5, 0, 9.0, 1.0, False, ()),
                (6, 0, 13.0, 0.0, False, ()),
            ],
            [
                (6, 0, 13.0, 0.0, False, ()),
                (7, 0, 0.0, 0.0, False, ()),
                (1, 0, 2.0, 1.0, True, ()),
            ],
            [
                (1, 0, 2.0, 1.0, True, ()),
                (2, 0, 3.0, 1.0, False, ()),
                (3, 0, 5.0, 1.0, False, ()),
            ],
            [
                (3, 0, 5.0, 1.0, False, ()),
                (4, 0, 7.0, 1.0, False, ()),
                (5, 0, 9.0, 1.0, False, ()),
            ],
            [
                (5, 0, 9.0, 1.0, False, ()),
                (6, 0, 13.0, 0.0, False, ()),
                (7, 0, 0.0, 0.0, False, ()),
            ],
        ),
        end_behavior=adders.EndBehavior.CONTINUE,
        repeat_episode_times=3
    ),
    dict(
        testcase_name='NonBreakingSequenceMultipleTruncatedEpisodes',
        sequence_length=3,
        period=2,
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.transition(reward=3.0, observation=3)),
            (0, dm_env.transition(reward=5.0, observation=4)),
            (0, dm_env.transition(reward=7.0, observation=5)),
            (0, dm_env.transition(reward=9.0, observation=6)),
            (0, dm_env.truncation(reward=13.0, observation=7)),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode, extra)
            [
                (1, 0, 2.0, 1.0, True, ()),
                (2, 0, 3.0, 1.0, False, ()),
                (3, 0, 5.0, 1.0, False, ()),
            ],
            [
                (3, 0, 5.0, 1.0, False, ()),
                (4, 0, 7.0, 1.0, False, ()),
                (5, 0, 9.0, 1.0, False, ()),
            ],
            [
                (5, 0, 9.0, 1.0, False, ()),
                (6, 0, 13.0, 1.0, False, ()),
                (7, 0, 0.0, 0.0, False, ()),
            ],
            [
                (7, 0, 0.0, 0.0, False, ()),
                (1, 0, 2.0, 1.0, True, ()),
                (2, 0, 3.0, 1.0, False, ()),
            ],
            [
                (2, 0, 3.0, 1.0, False, ()),
                (3, 0, 5.0, 1.0, False, ()),
                (4, 0, 7.0, 1.0, False, ()),
            ],
            [
                (4, 0, 7.0, 1.0, False, ()),
                (5, 0, 9.0, 1.0, False, ()),
                (6, 0, 13.0, 1.0, False, ()),
            ],
            [
                (6, 0, 13.0, 1.0, False, ()),
                (7, 0, 0.0, 0.0, False, ()),
                (1, 0, 2.0, 1.0, True, ()),
            ],
            [
                (1, 0, 2.0, 1.0, True, ()),
                (2, 0, 3.0, 1.0, False, ()),
                (3, 0, 5.0, 1.0, False, ()),
            ],
            [
                (3, 0, 5.0, 1.0, False, ()),
                (4, 0, 7.0, 1.0, False, ()),
                (5, 0, 9.0, 1.0, False, ()),
            ],
            [
                (5, 0, 9.0, 1.0, False, ()),
                (6, 0, 13.0, 1.0, False, ()),
                (7, 0, 0.0, 0.0, False, (),)
            ],
        ),
        end_behavior=adders.EndBehavior.CONTINUE,
        repeat_episode_times=3
    ),
    dict(
        testcase_name='EndBehavior_WRITE',
        sequence_length=3,
        period=2,
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.transition(reward=3.0, observation=3)),
            (0, dm_env.transition(reward=5.0, observation=4)),
            (0, dm_env.transition(reward=7.0, observation=5)),
            (0, dm_env.termination(reward=8.0, observation=6)),
        ),
        expected_sequences=(
            # (observation, action, reward, discount, start_of_episode, extra)
            [
                (1, 0, 2.0, 1.0, True, ()),
                (2, 0, 3.0, 1.0, False, ()),
                (3, 0, 5.0, 1.0, False, ()),
            ],
            [
                (3, 0, 5.0, 1.0, False, ()),
                (4, 0, 7.0, 1.0, False, ()),
                (5, 0, 8.0, 0.0, False, ()),
            ],
            [
                (4, 0, 7.0, 1.0, False, ()),
                (5, 0, 8.0, 0.0, False, ()),
                (6, 0, 0.0, 0.0, False, ()),
            ],
        ),
        end_behavior=adders.EndBehavior.WRITE,
    ),
]


class SequenceAdderTest(test_utils.AdderTestMixin, parameterized.TestCase):

  @parameterized.named_parameters(*TEST_CASES)
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
        end_behavior=end_behavior)

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
