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

"""Tests for structured."""

from typing import Sequence

from acme import types
from acme.adders.reverb import sequence as adders
from acme.adders.reverb import structured
from acme.adders.reverb import test_cases
from acme.adders.reverb import test_utils
import dm_env
import numpy as np
from reverb import structured_writer as sw
import tree

from absl.testing import absltest
from absl.testing import parameterized


class StructuredAdderTest(test_utils.AdderTestMixin, parameterized.TestCase):

  @parameterized.named_parameters(*test_cases.BASE_TEST_CASES_FOR_SEQUENCE_ADDER
                                 )
  def test_sequence_adder(self,
                          sequence_length: int,
                          period: int,
                          first,
                          steps,
                          expected_sequences,
                          end_behavior: adders.EndBehavior,
                          repeat_episode_times: int = 1):

    env_spec, extras_spec = test_utils.get_specs(steps[0])
    step_spec = structured.create_step_spec(env_spec, extras_spec)

    should_pad_trajectory = end_behavior == adders.EndBehavior.ZERO_PAD

    def _maybe_zero_pad(flat_trajectory):
      trajectory = tree.unflatten_as(step_spec, flat_trajectory)

      if not should_pad_trajectory:
        return trajectory

      padding_length = sequence_length - flat_trajectory[0].shape[0]
      if padding_length == 0:
        return trajectory

      padding = tree.map_structure(
          lambda x: np.zeros([padding_length, *x.shape[1:]], x.dtype),
          trajectory)

      return tree.map_structure(lambda *x: np.concatenate(x), trajectory,
                                padding)

    # The StructuredAdder does not support adding padding steps as we assume
    # that the padding will be added on the learner side.
    if end_behavior == adders.EndBehavior.ZERO_PAD:
      end_behavior = adders.EndBehavior.TRUNCATE

    configs = structured.create_sequence_config(
        step_spec=step_spec,
        sequence_length=sequence_length,
        period=period,
        end_of_episode_behavior=end_behavior)
    adder = structured.StructuredAdder(
        client=self.client,
        max_in_flight_items=0,
        configs=configs,
        step_spec=step_spec)

    super().run_test_adder(
        adder=adder,
        first=first,
        steps=steps,
        expected_items=expected_sequences,
        repeat_episode_times=repeat_episode_times,
        end_behavior=end_behavior,
        item_transform=_maybe_zero_pad,
        signature=sw.infer_signature(configs, step_spec))

  @parameterized.named_parameters(*test_cases.TEST_CASES_FOR_TRANSITION_ADDER)
  def test_transition_adder(
      self,
      n_step: int,
      additional_discount: float,
      first: dm_env.TimeStep,
      steps: Sequence[dm_env.TimeStep],
      expected_transitions: Sequence[types.Transition],
  ):
    env_spec, extras_spec = test_utils.get_specs(steps[0])
    step_spec = structured.create_step_spec(env_spec, extras_spec)

    configs = structured.create_n_step_transition_config(
        step_spec=step_spec, n_step=n_step)

    adder = structured.StructuredAdder(
        client=self.client,
        max_in_flight_items=0,
        configs=configs,
        step_spec=step_spec)

    def n_step_from_trajectory(trajectory: Sequence[types.Transition]):
      trajectory = tree.unflatten_as(step_spec, trajectory)
      return structured.n_step_from_trajectory(trajectory, additional_discount)

    super().run_test_adder(
        adder=adder,
        first=first,
        steps=steps,
        expected_items=expected_transitions,
        stack_sequence_fields=False,
        item_transform=n_step_from_trajectory,
        signature=sw.infer_signature(configs, step_spec),
    )


if __name__ == '__main__':
  absltest.main()
