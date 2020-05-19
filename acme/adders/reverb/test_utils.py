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

import dm_env


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
    self.priorities.append((table, self.timesteps[-num_timesteps:], priority))

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
