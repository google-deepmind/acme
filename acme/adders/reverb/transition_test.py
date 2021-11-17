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
from acme.adders.reverb import test_cases
from acme.adders.reverb import test_utils
from acme.adders.reverb import transition as adders


class NStepTransitionAdderTest(test_utils.AdderTestMixin,
                               parameterized.TestCase):

  @parameterized.named_parameters(*test_cases.TEST_CASES_FOR_TRANSITION_ADDER)
  def test_adder(self, n_step, additional_discount, first, steps,
                 expected_transitions):
    adder = adders.NStepTransitionAdder(self.client, n_step,
                                        additional_discount)
    super().run_test_adder(
        adder=adder,
        first=first,
        steps=steps,
        expected_items=expected_transitions,
        stack_sequence_fields=False,
        signature=adder.signature(*test_utils.get_specs(steps[0])))


if __name__ == '__main__':
  absltest.main()
