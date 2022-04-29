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

"""Tests for Episode adders."""

from acme.adders.reverb import episode as adders
from acme.adders.reverb import test_utils
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized


class EpisodeAdderTest(test_utils.AdderTestMixin, parameterized.TestCase):

  @parameterized.parameters(2, 10, 50)
  def test_adder(self, max_sequence_length):
    adder = adders.EpisodeAdder(self.client, max_sequence_length)

    # Create a simple trajectory to add.
    observations = range(max_sequence_length)
    first, steps = test_utils.make_trajectory(observations)

    expected_episode = test_utils.make_sequence(observations)
    self.run_test_adder(
        adder=adder,
        first=first,
        steps=steps,
        expected_items=[expected_episode],
        signature=adder.signature(*test_utils.get_specs(steps[0])))

  @parameterized.parameters(2, 10, 50)
  def test_max_sequence_length(self, max_sequence_length):
    adder = adders.EpisodeAdder(self.client, max_sequence_length)

    first, steps = test_utils.make_trajectory(range(max_sequence_length + 1))
    adder.add_first(first)
    for action, step in steps[:-1]:
      adder.add(action, step)

    # We should have max_sequence_length-1 timesteps that have been written,
    # where the -1 is due to the dangling observation (ie we have actually
    # seen max_sequence_length observations).
    self.assertEqual(self.num_items(), 0)

    # Adding one more step should raise an error.
    with self.assertRaises(ValueError):
      action, step = steps[-1]
      adder.add(action, step)

    # Since the last insert failed it should not affect the internal state.
    self.assertEqual(self.num_items(), 0)

  @parameterized.parameters((2, 1), (10, 2), (50, 5))
  def test_padding(self, max_sequence_length, padding):
    adder = adders.EpisodeAdder(
        self.client,
        max_sequence_length + padding,
        padding_fn=np.zeros)

    # Create a simple trajectory to add.
    observations = range(max_sequence_length)
    first, steps = test_utils.make_trajectory(observations)

    expected_episode = test_utils.make_sequence(observations)
    for _ in range(padding):
      expected_episode.append((0, 0, 0.0, 0.0, False, ()))

    self.run_test_adder(
        adder=adder,
        first=first,
        steps=steps,
        expected_items=[expected_episode],
        signature=adder.signature(*test_utils.get_specs(steps[0])))

  @parameterized.parameters((2, 1), (10, 2), (50, 5))
  def test_nonzero_padding(self, max_sequence_length, padding):
    adder = adders.EpisodeAdder(
        self.client,
        max_sequence_length + padding,
        padding_fn=lambda s, d: np.zeros(s, d) - 1)

    # Create a simple trajectory to add.
    observations = range(max_sequence_length)
    first, steps = test_utils.make_trajectory(observations)

    expected_episode = test_utils.make_sequence(observations)
    for _ in range(padding):
      expected_episode.append((-1, -1, -1.0, -1.0, False, ()))

    self.run_test_adder(
        adder=adder,
        first=first,
        steps=steps,
        expected_items=[expected_episode],
        signature=adder.signature(*test_utils.get_specs(steps[0])))


if __name__ == '__main__':
  absltest.main()
