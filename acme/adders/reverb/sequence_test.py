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
import numpy as np

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
]


class SequenceAdderTest(parameterized.TestCase):

  @parameterized.named_parameters(*TEST_CASES)
  def test_adder(self, sequence_length: int, period: int, first, steps,
                 expected_sequences):
    client = test_utils.FakeClient()
    adder = adders.SequenceAdder(
        client, sequence_length=sequence_length, period=period)

    # Add all the data up to the final step.
    adder.add_first(first)
    for step in steps[:-1]:
      adder.add(*step)

    # Make sure the writer has been created but not closed.
    self.assertLen(client.writers, 1)
    self.assertFalse(client.writers[0].closed)

    # Add the final step.
    adder.add(*steps[-1])

    # Ending the episode should close the writer. No new writer should yet have
    # been created as it is constructed lazily.
    self.assertLen(client.writers, 1)
    self.assertTrue(client.writers[0].closed)

    # Make sure our expected and observed transitions match.
    observed_sequences = list(p[1] for p in client.writers[0].priorities)
    for exp, obs in zip(expected_sequences, observed_sequences):
      np.testing.assert_array_equal(exp, obs)

    # Add the start of a second trajectory.
    adder.add_first(first)
    adder.add(*steps[0])

    # Make sure this creates an open writer.
    self.assertLen(client.writers, 2)
    self.assertFalse(client.writers[1].closed)


if __name__ == '__main__':
  absltest.main()
