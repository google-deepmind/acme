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

"""Tests for NStepTransition adders."""

from absl.testing import absltest
from absl.testing import parameterized
from acme.adders.reverb import test_utils
from acme.adders.reverb import transition as adders
import dm_env

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
            (1, 0, 0.0, 1.0, 2, ()),
            (2, 0, 0.0, 1.0, 3, ()),
            (3, 0, 1.0, 0.0, 4, ()),
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
            ({'foo': 1}, 0, 0.0, 1.0, {'foo': 2}, ()),
            ({'foo': 2}, 0, 0.0, 1.0, {'foo': 3}, ()),
            ({'foo': 3}, 0, 1.0, 0.0, {'foo': 4}, ()),
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
            (1, 0, 1.0, 0.50, 2, ()),
            (1, 0, 1.5, 0.25, 3, ()),
            (2, 0, 1.5, 0.00, 4, ()),
            (3, 0, 1.0, 0.00, 4, ()),
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
            (1, 0, 1.00, 0.5, 2, ()),
            (1, 0, 1.20, 0.1, 3, ()),
            (1, 0, 1.24, 0.0, 4, ()),
            (2, 0, 1.20, 0.0, 4, ()),
            (3, 0, 1.00, 0.0, 4, ()),
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
            (1, 0, 2, 1.00, 2, ()),
            (1, 0, 2 + 0.5 * 3, 0.50, 3, ()),
            (1, 0, 2 + 0.5 * 3 + 0.25 * 5, 0.25, 4, ()),
            (2, 0, 3 + 0.5 * 5 + 0.25 * 7, 0.00, 5, ()),
            (3, 0, 5 + 0.5 * 7, 0.00, 5, ()),
            (4, 0, 7, 0.00, 5, ()),
        ))
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
        expected_items=expected_transitions)


if __name__ == '__main__':
  absltest.main()
