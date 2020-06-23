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
            # (observation, action, reward, discount, extra)
            [(1, 0, 2.0, 1.0, ()), (2, 0, 3.0, 1.0, ()), (3, 0, 5.0, 1.0, ())],
            [(2, 0, 3.0, 1.0, ()), (3, 0, 5.0, 1.0, ()), (4, 0, 7.0, 0.0, ())],
            [(3, 0, 5.0, 1.0, ()), (4, 0, 7.0, 0.0, ()), (5, 0, 0.0, 0.0, ())],
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
            # (observation, action, reward, discount, extra)
            [(1, 0, 2.0, 1.0, ()), (2, 0, 3.0, 1.0, ()), (3, 0, 5.0, 1.0, ())],
            [(3, 0, 5.0, 1.0, ()), (4, 0, 7.0, 0.0, ()), (5, 0, 0.0, 0.0, ())],
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
            # (observation, action, reward, discount, extra)
            [(1, 0, 2.0, 1.0, ()), (2, 0, 3.0, 0.0, ()),
             (3, 0, 0.0, 0.0, ())],),
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
            # (observation, action, reward, discount, extra)
            [(1, 0, 2.0, 1.0, ()), (2, 0, 3.0, 0.0, ()),
             (3, 0, 0.0, 0.0, ())],),
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
            # (observation, action, reward, discount, extra)
            [
                (1, 0, 2.0, 1.0, ()),
                (2, 0, 3.0, 0.0, ()),
                (3, 0, 0.0, 0.0, ()),
                (0, 0, 0.0, 0.0, ()),
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
            # (observation, action, reward, discount, extra)
            [
                (1, 0, 2.0, 1.0, ()),
                (2, 0, 3.0, 0.0, ()),
                (3, 0, 0.0, 0.0, ()),
                (0, 0, 0.0, 0.0, ()),
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
            # (observation, action, reward, discount, extra)
            [
                (1, 0, 2.0, 1.0, ()),
                (2, 0, 3.0, 0.0, ()),
                (3, 0, 0.0, 0.0, ()),
            ],),
        pad_end_of_episode=False,
    ),
]


class SequenceAdderTest(test_utils.AdderTestMixin, parameterized.TestCase):

  @parameterized.named_parameters(*TEST_CASES)
  def test_adder(self, sequence_length: int, period: int, first, steps,
                 expected_sequences, pad_end_of_episode: bool = True):
    adder = adders.SequenceAdder(
        self.client,
        sequence_length=sequence_length,
        period=period,
        pad_end_of_episode=pad_end_of_episode)
    super().run_test_adder(
        adder=adder,
        first=first,
        steps=steps,
        expected_items=expected_sequences)


if __name__ == '__main__':
  absltest.main()
