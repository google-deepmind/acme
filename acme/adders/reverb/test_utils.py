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

"""Utilities for testing Reverb adders."""

from typing import Any, Sequence, Tuple, Union

from absl.testing import absltest
from acme import specs
from acme.adders.reverb import base
import dm_env
import numpy as np
import tensorflow as tf
import tree


class FakeWriter:
  """Fake writer for testing."""

  def __init__(self,
               max_sequence_length,
               delta_encoded=False,
               chunk_length=None):
    self.max_sequence_length = max_sequence_length
    self.delta_encoded = delta_encoded
    self.chunk_length = chunk_length

    self.timesteps = []
    self.priorities = []
    self.closed = False

  def append(self, timestep):
    assert not self.closed, 'Trying to use closed Writer'
    self.timesteps.append(timestep)

  def create_item(self, table, num_timesteps, priority):
    assert not self.closed, 'Trying to use closed Writer'
    assert num_timesteps <= len(self.timesteps)
    assert num_timesteps <= self.max_sequence_length
    item = self.timesteps[-num_timesteps:]
    if num_timesteps == 1:
      item = item[0]
    self.priorities.append((table, item, priority))

  def close(self):
    assert not self.closed, 'Trying to use closed Writer'
    self.closed = True


class FakeClient:
  """Fake client for testing."""

  def __init__(self):
    self.writers = []

  def writer(self, max_sequence_length, delta_encoded=False, chunk_length=None):
    new_writer = FakeWriter(max_sequence_length, delta_encoded, chunk_length)
    self.writers.append(new_writer)
    return new_writer


def make_trajectory(observations):
  """Make a simple trajectory from a sequence of observations.

  Arguments:
    observations: a sequence of observations.

  Returns:
    a tuple (first, steps) where first contains the initial dm_env.TimeStep
    object and steps contains a list of (action, step) tuples. The length of
    steps is given by episode_length.
  """
  first = dm_env.restart(observations[0])
  middle = [(0, dm_env.transition(reward=0.0, observation=observation))
            for observation in observations[1:-1]]
  last = (0, dm_env.termination(reward=0.0, observation=observations[-1]))
  return first, middle + [last]


def make_sequence(observations):
  """Create a sequence of timesteps of the form `first, [second, ..., last]`."""
  first, steps = make_trajectory(observations)
  observation = first.observation
  sequence = []
  for action, timestep in steps:
    extras = ()
    sequence.append(
        (observation, action, timestep.reward, timestep.discount, extras))
    observation = timestep.observation
  sequence.append((observation, 0, 0.0, 0.0, ()))
  return sequence


def _numeric_to_spec(x: Union[float, int, np.ndarray]):
  if isinstance(x, np.ndarray):
    return specs.Array(shape=x.shape, dtype=x.dtype)
  elif isinstance(x, (float, int)):
    return specs.Array(shape=(), dtype=type(x))
  else:
    raise ValueError(f'Unsupported numeric: {type(x)}')


class AdderTestMixin(absltest.TestCase):
  """A helper mixin for testing Reverb adders.

  Note that any test inheriting from this mixin must also inherit from something
  that provides the Python unittest assert methods.
  """

  client: FakeClient

  def setUp(self):
    super().setUp()
    self.client = FakeClient()

  def run_test_adder(self,
                     adder: base.ReverbAdder,
                     first: dm_env.TimeStep,
                     steps: Sequence[Tuple[Any, dm_env.TimeStep]],
                     expected_items: Sequence[Any]):
    """Runs a unit test case for the adder.

    Args:
      adder: The instance of `base.ReverbAdder` that is being tested.
      first: The first `dm_env.TimeStep` that is used to call
        `base.ReverbAdder.add_first()`.
      steps: A sequence of (action, timestep) tuples that are passed to
        `base.ReverbAdder.add()`.
      expected_items: The sequence of items that are expected to be created
        by calling the adder's `add_first()` method on `first` and `add()` on
        all of the elements in `steps`.
    """
    if not steps:
      raise ValueError('At least one step must be given.')

    env_spec = tree.map_structure(
        _numeric_to_spec,
        specs.EnvironmentSpec(
            observations=steps[0][1].observation,
            actions=steps[0][0],
            rewards=steps[0][1].reward,
            discounts=steps[0][1].discount))
    signature = adder.signature(env_spec)

    # Add all the data up to the final step.
    adder.add_first(first)
    for action, ts in steps[:-1]:
      adder.add(action, next_timestep=ts)

    if len(steps) == 1:
      # adder.add() has not been called yet, so no writers have been created.
      self.assertEmpty(self.client.writers)
    else:
      # Make sure the writer has been created but not closed.
      self.assertLen(self.client.writers, 1)
      self.assertFalse(self.client.writers[0].closed)

    # Add the final step.
    adder.add(*steps[-1])

    # Ending the episode should close the writer. No new writer should yet have
    # been created as it is constructed lazily.
    self.assertLen(self.client.writers, 1)
    self.assertTrue(self.client.writers[0].closed)

    # Make sure our expected and observed data match.
    observed_items = [p[1] for p in self.client.writers[0].priorities]
    for expected_item, observed_item in zip(expected_items, observed_items):
      # Set check_types=False because
      tree.map_structure(
          np.testing.assert_array_almost_equal,
          expected_item,
          observed_item,
          check_types=False)

    def _check_signature(spec: tf.TensorSpec, value):
      # Convert int/float to numpy arrays of dtype np.int64 and np.float64.
      value = np.asarray(value)
      self.assertTrue(spec.is_compatible_with(tf.convert_to_tensor(value)))

    for step in self.client.writers[0].timesteps:
      tree.map_structure(_check_signature, signature, step)

    # Add the start of a second trajectory.
    adder.add_first(first)
    adder.add(*steps[0])

    # Make sure this creates an new writer.
    self.assertLen(self.client.writers, 2)
    # The writer is closed if the recently added `dm_env.TimeStep`'s' step_type
    # is `dm_env.StepType.LAST`.
    if steps[0][1].last():
      self.assertTrue(self.client.writers[1].closed)
    else:
      self.assertFalse(self.client.writers[1].closed)
