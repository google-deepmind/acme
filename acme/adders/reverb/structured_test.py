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
from acme.utils import tree_utils
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
  def test_transition_adder(self, n_step: int, additional_discount: float,
                            first: dm_env.TimeStep,
                            steps: Sequence[dm_env.TimeStep],
                            expected_transitions: Sequence[types.Transition]):

    env_spec, extras_spec = test_utils.get_specs(steps[0])
    step_spec = structured.create_step_spec(env_spec, extras_spec)

    def _as_n_step_transition(flat_trajectory):
      trajectory = tree.unflatten_as(step_spec, flat_trajectory)

      rewards, discount = _compute_cumulative_quantities(
          rewards=trajectory.reward,
          discounts=trajectory.discount,
          additional_discount=additional_discount,
          n_step=tree.flatten(trajectory.reward)[0].shape[0])

      tmap = tree.map_structure
      return types.Transition(
          observation=tmap(lambda x: x[0], trajectory.observation),
          action=tmap(lambda x: x[0], trajectory.action),
          reward=rewards,
          discount=discount,
          next_observation=tmap(lambda x: x[-1], trajectory.observation),
          extras=tmap(lambda x: x[0], trajectory.extras))

    configs = structured.create_n_step_transition_config(
        step_spec=step_spec, n_step=n_step)

    adder = structured.StructuredAdder(
        client=self.client,
        max_in_flight_items=0,
        configs=configs,
        step_spec=step_spec)

    super().run_test_adder(
        adder=adder,
        first=first,
        steps=steps,
        expected_items=expected_transitions,
        stack_sequence_fields=False,
        item_transform=_as_n_step_transition,
        signature=sw.infer_signature(configs, step_spec))


def _compute_cumulative_quantities(rewards: types.NestedArray,
                                   discounts: types.NestedArray,
                                   additional_discount: float, n_step: int):
  """Stolen from TransitionAdder."""

  # Give the same tree structure to the n-step return accumulator,
  # n-step discount accumulator, and self.discount, so that they can be
  # iterated in parallel using tree.map_structure.
  rewards, discounts, self_discount = tree_utils.broadcast_structures(
      rewards, discounts, additional_discount)
  flat_rewards = tree.flatten(rewards)
  flat_discounts = tree.flatten(discounts)
  flat_self_discount = tree.flatten(self_discount)

  # Copy total_discount as it is otherwise read-only.
  total_discount = [np.copy(a[0]) for a in flat_discounts]

  # Broadcast n_step_return to have the broadcasted shape of
  # reward * discount.
  n_step_return = [
      np.copy(np.broadcast_to(r[0],
                              np.broadcast(r[0], d).shape))
      for r, d in zip(flat_rewards, total_discount)
  ]

  # NOTE: total_discount will have one less self_discount applied to it than
  # the value of self._n_step. This is so that when the learner/update uses
  # an additional discount we don't apply it twice. Inside the following loop
  # we will apply this right before summing up the n_step_return.
  for i in range(1, n_step):
    for nsr, td, r, d, sd in zip(n_step_return, total_discount, flat_rewards,
                                 flat_discounts, flat_self_discount):
      # Equivalent to: `total_discount *= self._discount`.
      td *= sd
      # Equivalent to: `n_step_return += reward[i] * total_discount`.
      nsr += r[i] * td
      # Equivalent to: `total_discount *= discount[i]`.
      td *= d[i]

  n_step_return = tree.unflatten_as(rewards, n_step_return)
  total_discount = tree.unflatten_as(rewards, total_discount)
  return n_step_return, total_discount


if __name__ == '__main__':
  absltest.main()
