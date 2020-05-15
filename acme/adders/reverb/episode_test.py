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

"""Tests for Episode adders."""

from absl.testing import absltest
from absl.testing import parameterized

from acme.adders.reverb import base
from acme.adders.reverb import episode as adders
from acme.adders.reverb import test_utils


class EpisodeAdderTest(parameterized.TestCase):

  @parameterized.parameters(2, 10, 50)
  def test_adder(self, max_sequence_length):
    client = test_utils.FakeClient()
    adder = adders.EpisodeAdder(client, max_sequence_length)

    # Create a simple trajectory to add.
    observations = range(max_sequence_length)
    first, steps = test_utils.make_trajectory(observations)

    # Add everything up to the final transition.
    adder.add_first(first)
    for action, step in steps[:-1]:
      adder.add(action, step)

    if max_sequence_length == 2:
      # Nothing has been written since we only have an initial step and a
      # final step (corresponding to a single transition).
      self.assertEmpty(client.writers)

    else:
      # No priorities should have been written so far but all timesteps (all
      # but the last one) should have been sent to the writer.
      self.assertEmpty(client.writers[0].priorities)
      self.assertLen(client.writers[0].timesteps, max_sequence_length - 2)

    # Adding a terminal timestep should close the writer and insert a
    # prioritized sample referencing all the timesteps (including the padded
    # terminating observation).
    action, step = steps[-1]
    adder.add(action, step)

    # The writer should be closed and should have max_sequence_length timesteps.
    self.assertTrue(client.writers[0].closed)
    self.assertLen(client.writers[0].timesteps, max_sequence_length)

    # Make the sequence of data and the priority table entry we expect.
    expected_sequence = test_utils.make_sequence(observations)
    expected_entry = (base.DEFAULT_PRIORITY_TABLE, expected_sequence, 1.0)

    self.assertEqual(client.writers[0].priorities, [expected_entry])

  @parameterized.parameters(2, 10, 50)
  def test_max_sequence_length(self, max_sequence_length):
    client = test_utils.FakeClient()
    adder = adders.EpisodeAdder(client, max_sequence_length)

    first, steps = test_utils.make_trajectory(range(max_sequence_length + 1))
    adder.add_first(first)
    for action, step in steps[:-1]:
      adder.add(action, step)

    # We should have max_sequence_length-1 timesteps that have been written,
    # where the -1 is due to the dangling observation (ie we have actually
    # seen max_sequence_length observations).
    self.assertLen(client.writers[0].timesteps, max_sequence_length - 1)

    # Adding one more step should raise an error.
    with self.assertRaises(ValueError):
      action, step = steps[-1]
      adder.add(action, step)

    # Since the last insert failed it should not affect the internal state.
    self.assertLen(client.writers[0].timesteps, max_sequence_length - 1)

  def test_delta_encoded_and_chunk_length(self):
    max_sequence_length = 5
    client = test_utils.FakeClient()
    adder = adders.EpisodeAdder(
        client,
        max_sequence_length,
        delta_encoded=True,
        chunk_length=max_sequence_length)

    # Add an episode.
    first, steps = test_utils.make_trajectory(range(max_sequence_length))
    adder.add_first(first)
    for action, step in steps:
      adder.add(action, step)

    self.assertTrue(client.writers[0].delta_encoded)
    self.assertEqual(max_sequence_length, client.writers[0].chunk_length)


if __name__ == '__main__':
  absltest.main()
