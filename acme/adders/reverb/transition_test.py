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

"""Tests for NStepTransition adders."""

from absl.testing import absltest
from absl.testing import parameterized
from acme import types
from acme.adders.reverb import test_utils
from acme.adders.reverb import transition as adders
import dm_env
import numpy as np

# Define the main set of test cases; these are given as parameterized tests to
# the test_adder method and describe a trajectory to add to replay and the
# expected transitions that should result from this trajectory. The expected
# transitions are of the form: (observation, action, reward, discount,
# next_observation, extras).
TEST_CASES = [
    dict(
        testcase_name='OneStepFinalReward',
        n_step=1,
        additional_discount=1.0,
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=0.0, observation=2)),
            (0, dm_env.transition(reward=0.0, observation=3)),
            (0, dm_env.termination(reward=1.0, observation=4)),
        ),
        expected_transitions=(
            types.Transition(1, 0, 0.0, 1.0, 2),
            types.Transition(2, 0, 0.0, 1.0, 3),
            types.Transition(3, 0, 1.0, 0.0, 4),
        )),
    dict(
        testcase_name='OneStepDict',
        n_step=1,
        additional_discount=1.0,
        first=dm_env.restart({'foo': 1}),
        steps=(
            (0, dm_env.transition(reward=0.0, observation={'foo': 2})),
            (0, dm_env.transition(reward=0.0, observation={'foo': 3})),
            (0, dm_env.termination(reward=1.0, observation={'foo': 4})),
        ),
        expected_transitions=(
            types.Transition({'foo': 1}, 0, 0.0, 1.0, {'foo': 2}),
            types.Transition({'foo': 2}, 0, 0.0, 1.0, {'foo': 3}),
            types.Transition({'foo': 3}, 0, 1.0, 0.0, {'foo': 4}),
        )),
    dict(
        testcase_name='OneStepExtras',
        n_step=1,
        additional_discount=1.0,
        first=dm_env.restart(1),
        steps=(
            (
                0,
                dm_env.transition(reward=0.0, observation=2),
                {
                    'state': 0
                },
            ),
            (
                0,
                dm_env.transition(reward=0.0, observation=3),
                {
                    'state': 1
                },
            ),
            (
                0,
                dm_env.termination(reward=1.0, observation=4),
                {
                    'state': 2
                },
            ),
        ),
        expected_transitions=(
            types.Transition(1, 0, 0.0, 1.0, 2, {'state': 0}),
            types.Transition(2, 0, 0.0, 1.0, 3, {'state': 1}),
            types.Transition(3, 0, 1.0, 0.0, 4, {'state': 2}),
        )),
    dict(
        testcase_name='OneStepExtrasZeroes',
        n_step=1,
        additional_discount=1.0,
        first=dm_env.restart(1),
        steps=(
            (
                0,
                dm_env.transition(reward=0.0, observation=2),
                np.zeros(1),
            ),
            (
                0,
                dm_env.transition(reward=0.0, observation=3),
                np.zeros(1),
            ),
            (
                0,
                dm_env.termination(reward=1.0, observation=4),
                np.zeros(1),
            ),
        ),
        expected_transitions=(
            types.Transition(1, 0, 0.0, 1.0, 2, np.zeros(1)),
            types.Transition(2, 0, 0.0, 1.0, 3, np.zeros(1)),
            types.Transition(3, 0, 1.0, 0.0, 4, np.zeros(1)),
        )),
    dict(
        testcase_name='TwoStep',
        n_step=2,
        additional_discount=1.0,
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=1.0, observation=2, discount=0.5)),
            (0, dm_env.transition(reward=1.0, observation=3, discount=0.5)),
            (0, dm_env.termination(reward=1.0, observation=4)),
        ),
        expected_transitions=(
            types.Transition(1, 0, 1.0, 0.50, 2),
            types.Transition(1, 0, 1.5, 0.25, 3),
            types.Transition(2, 0, 1.5, 0.00, 4),
            types.Transition(3, 0, 1.0, 0.00, 4),
        )),
    dict(
        testcase_name='TwoStepStructuredReward',
        n_step=2,
        additional_discount=1.0,
        first=dm_env.restart(1),
        steps=(
            (0,
             dm_env.transition(reward=(1.0, 2.0), observation=2, discount=0.5)),
            (0,
             dm_env.transition(reward=(1.0, 2.0), observation=3, discount=0.5)),
            (0, dm_env.termination(reward=(1.0, 2.0), observation=4)),
        ),
        expected_transitions=(
            types.Transition(1, 0, (1.0, 2.0), (0.50, 0.50), 2),
            types.Transition(1, 0, (1.5, 3.0), (0.25, 0.25), 3),
            types.Transition(2, 0, (1.5, 3.0), (0.00, 0.00), 4),
            types.Transition(3, 0, (1.0, 2.0), (0.00, 0.00), 4),
        )),
    dict(
        testcase_name='TwoStepNDArrayReward',
        n_step=2,
        additional_discount=1.0,
        first=dm_env.restart(1),
        steps=(
            (0,
             dm_env.transition(
                 reward=np.array((1.0, 2.0)), observation=2, discount=0.5)),
            (0,
             dm_env.transition(
                 reward=np.array((1.0, 2.0)), observation=3, discount=0.5)),
            (0, dm_env.termination(reward=np.array((1.0, 2.0)), observation=4)),
        ),
        expected_transitions=(
            types.Transition(1, 0, np.array((1.0, 2.0)), np.array((0.50, 0.50)),
                             2),
            types.Transition(1, 0, np.array((1.5, 3.0)), np.array((0.25, 0.25)),
                             3),
            types.Transition(2, 0, np.array((1.5, 3.0)), np.array((0.00, 0.00)),
                             4),
            types.Transition(3, 0, np.array((1.0, 2.0)), np.array((0.00, 0.00)),
                             4),
        )),
    dict(
        testcase_name='TwoStepStructuredDiscount',
        n_step=2,
        additional_discount=1.0,
        first=dm_env.restart(1),
        steps=(
            (0,
             dm_env.transition(
                 reward=1.0, observation=2, discount={
                     'a': 0.5,
                     'b': 0.1
                 })),
            (0,
             dm_env.transition(
                 reward=1.0, observation=3, discount={
                     'a': 0.5,
                     'b': 0.1
                 })),
            (0, dm_env.termination(reward=1.0,
                                   observation=4)._replace(discount={
                                       'a': 0.0,
                                       'b': 0.0
                                   })),
        ),
        expected_transitions=(
            types.Transition(1, 0, {
                'a': 1.0,
                'b': 1.0
            }, {
                'a': 0.50,
                'b': 0.10
            }, 2),
            types.Transition(1, 0, {
                'a': 1.5,
                'b': 1.1
            }, {
                'a': 0.25,
                'b': 0.01
            }, 3),
            types.Transition(2, 0, {
                'a': 1.5,
                'b': 1.1
            }, {
                'a': 0.00,
                'b': 0.00
            }, 4),
            types.Transition(3, 0, {
                'a': 1.0,
                'b': 1.0
            }, {
                'a': 0.00,
                'b': 0.00
            }, 4),
        )),
    dict(
        testcase_name='TwoStepNDArrayDiscount',
        n_step=2,
        additional_discount=1.0,
        first=dm_env.restart(1),
        steps=(
            (0,
             dm_env.transition(
                 reward=1.0, observation=2, discount=np.array((0.5, 0.1)))),
            (0,
             dm_env.transition(
                 reward=1.0, observation=3, discount=np.array((0.5, 0.1)))),
            (0, dm_env.termination(
                reward=1.0,
                observation=4)._replace(discount=np.array((0.0, 0.0)))),
        ),
        expected_transitions=(
            types.Transition(1, 0, np.array((1.0, 1.0)), np.array((0.50, 0.10)),
                             2),
            types.Transition(1, 0, np.array((1.5, 1.1)), np.array((0.25, 0.01)),
                             3),
            types.Transition(2, 0, np.array((1.5, 1.1)), np.array((0.00, 0.00)),
                             4),
            types.Transition(3, 0, np.array((1.0, 1.0)), np.array((0.00, 0.00)),
                             4),
        )),
    dict(
        testcase_name='TwoStepBroadcastedNDArrays',
        n_step=2,
        additional_discount=1.0,
        first=dm_env.restart(1),
        steps=(
            (0,
             dm_env.transition(
                 reward=np.array([[1.0, 2.0]]),
                 observation=2,
                 discount=np.array([[0.5], [0.1]]))),
            (0,
             dm_env.transition(
                 reward=np.array([[1.0, 2.0]]),
                 observation=3,
                 discount=np.array([[0.5], [0.1]]))),
            (0, dm_env.termination(
                reward=np.array([[1.0, 2.0]]),
                observation=4)._replace(discount=np.array([[0.0], [0.0]]))),
        ),
        expected_transitions=(
            types.Transition(1, 0, np.array([[1.0, 2.0], [1.0, 2.0]]),
                             np.array([[0.50], [0.10]]), 2),
            types.Transition(1, 0, np.array([[1.5, 3.0], [1.1, 2.2]]),
                             np.array([[0.25], [0.01]]), 3),
            types.Transition(2, 0, np.array([[1.5, 3.0], [1.1, 2.2]]),
                             np.array([[0.00], [0.00]]), 4),
            types.Transition(3, 0, np.array([[1.0, 2.0], [1.0, 2.0]]),
                             np.array([[0.00], [0.00]]), 4),
        )),
    dict(
        testcase_name='TwoStepStructuredBroadcastedNDArrays',
        n_step=2,
        additional_discount=1.0,
        first=dm_env.restart(1),
        steps=(
            (0,
             dm_env.transition(
                 reward={'a': np.array([[1.0, 2.0]])},
                 observation=2,
                 discount=np.array([[0.5], [0.1]]))),
            (0,
             dm_env.transition(
                 reward={'a': np.array([[1.0, 2.0]])},
                 observation=3,
                 discount=np.array([[0.5], [0.1]]))),
            (0,
             dm_env.termination(
                 reward={
                     'a': np.array([[1.0, 2.0]])
                 }, observation=4)._replace(discount=np.array([[0.0], [0.0]]))),
        ),
        expected_transitions=(
            types.Transition(1, 0, {'a': np.array([[1.0, 2.0], [1.0, 2.0]])},
                             {'a': np.array([[0.50], [0.10]])}, 2),
            types.Transition(1, 0, {'a': np.array([[1.5, 3.0], [1.1, 2.2]])},
                             {'a': np.array([[0.25], [0.01]])}, 3),
            types.Transition(2, 0, {'a': np.array([[1.5, 3.0], [1.1, 2.2]])},
                             {'a': np.array([[0.00], [0.00]])}, 4),
            types.Transition(3, 0, {'a': np.array([[1.0, 2.0], [1.0, 2.0]])},
                             {'a': np.array([[0.00], [0.00]])}, 4),
        )),
    dict(
        testcase_name='TwoStepWithExtras',
        n_step=2,
        additional_discount=1.0,
        first=dm_env.restart(1),
        steps=(
            (
                0,
                dm_env.transition(reward=1.0, observation=2, discount=0.5),
                {
                    'state': 0
                },
            ),
            (
                0,
                dm_env.transition(reward=1.0, observation=3, discount=0.5),
                {
                    'state': 1
                },
            ),
            (
                0,
                dm_env.termination(reward=1.0, observation=4),
                {
                    'state': 2
                },
            ),
        ),
        expected_transitions=(
            types.Transition(1, 0, 1.0, 0.50, 2, {'state': 0}),
            types.Transition(1, 0, 1.5, 0.25, 3, {'state': 0}),
            types.Transition(2, 0, 1.5, 0.00, 4, {'state': 1}),
            types.Transition(3, 0, 1.0, 0.00, 4, {'state': 2}),
        )),
    dict(
        testcase_name='ThreeStepDiscounted',
        n_step=3,
        additional_discount=0.4,
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=1.0, observation=2, discount=0.5)),
            (0, dm_env.transition(reward=1.0, observation=3, discount=0.5)),
            (0, dm_env.termination(reward=1.0, observation=4)),
        ),
        expected_transitions=(
            types.Transition(1, 0, 1.00, 0.5, 2),
            types.Transition(1, 0, 1.20, 0.1, 3),
            types.Transition(1, 0, 1.24, 0.0, 4),
            types.Transition(2, 0, 1.20, 0.0, 4),
            types.Transition(3, 0, 1.00, 0.0, 4),
        )),
    dict(
        testcase_name='ThreeStepVaryingReward',
        n_step=3,
        additional_discount=0.5,
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=2.0, observation=2)),
            (0, dm_env.transition(reward=3.0, observation=3)),
            (0, dm_env.transition(reward=5.0, observation=4)),
            (0, dm_env.termination(reward=7.0, observation=5)),
        ),
        expected_transitions=(
            types.Transition(1, 0, 2, 1.00, 2),
            types.Transition(1, 0, 2 + 0.5 * 3, 0.50, 3),
            types.Transition(1, 0, 2 + 0.5 * 3 + 0.25 * 5, 0.25, 4),
            types.Transition(2, 0, 3 + 0.5 * 5 + 0.25 * 7, 0.00, 5),
            types.Transition(3, 0, 5 + 0.5 * 7, 0.00, 5),
            types.Transition(4, 0, 7, 0.00, 5),
        )),
    dict(
        testcase_name='SingleTransitionEpisode',
        n_step=4,
        additional_discount=1.0,
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.termination(reward=1.0, observation=2)),
        ),
        expected_transitions=(
            types.Transition(1, 0, 1.00, 0.0, 2),
        )),
    dict(
        testcase_name='EpisodeShorterThanN',
        n_step=4,
        additional_discount=1.0,
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=1.0, observation=2)),
            (0, dm_env.termination(reward=1.0, observation=3)),
        ),
        expected_transitions=(
            types.Transition(1, 0, 1.00, 1.0, 2),
            types.Transition(1, 0, 2.00, 0.0, 3),
            types.Transition(2, 0, 1.00, 0.0, 3),
        )),
    dict(
        testcase_name='EpisodeEqualToN',
        n_step=3,
        additional_discount=1.0,
        first=dm_env.restart(1),
        steps=(
            (0, dm_env.transition(reward=1.0, observation=2)),
            (0, dm_env.termination(reward=1.0, observation=3)),
        ),
        expected_transitions=(
            types.Transition(1, 0, 1.00, 1.0, 2),
            types.Transition(1, 0, 2.00, 0.0, 3),
            types.Transition(2, 0, 1.00, 0.0, 3),
        )),
]


class NStepTransitionAdderTest(test_utils.AdderTestMixin,
                               parameterized.TestCase):

  @parameterized.named_parameters(*TEST_CASES)
  def test_adder(self, n_step, additional_discount, first, steps,
                 expected_transitions):
    adder = adders.NStepTransitionAdder(self.client, n_step,
                                        additional_discount)
    super().run_test_adder(
        adder=adder,
        first=first,
        steps=steps,
        expected_items=expected_transitions,
        stack_sequence_fields=False)


if __name__ == '__main__':
  absltest.main()
