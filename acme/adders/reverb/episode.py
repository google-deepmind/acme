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

"""Episode adders.

This implements full episode adders, potentially with padding.
"""

from typing import Optional

from acme import types
from acme.adders.reverb import base
from acme.adders.reverb import utils

import dm_env
import reverb


class EpisodeAdder(base.ReverbAdder):
  """Adder which adds entire episodes as trajectories."""

  def __init__(
      self,
      client: reverb.Client,
      max_sequence_length: int,
      delta_encoded: bool = False,
      chunk_length: Optional[int] = None,
      priority_fns: Optional[base.PriorityFnMapping] = None,
  ):
    super().__init__(
        client=client,
        buffer_size=max_sequence_length - 1,
        max_sequence_length=max_sequence_length,
        delta_encoded=delta_encoded,
        chunk_length=chunk_length,
        priority_fns=priority_fns,
    )

  def add(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
      extras: types.NestedArray = (),
  ):
    if len(self._buffer) == self._buffer.maxlen:
      # If the buffer is full that means we've buffered max_sequence_length-1
      # steps, one dangling observation, and are trying to add one more (which
      # will overflow the buffer).
      raise ValueError(
          'The number of observations within the same episode exceeds '
          'max_sequence_length')

    super().add(action, next_timestep, extras)

  def _write(self):
    # Append the previous step.
    self._writer.append(self._buffer[-1])

  def _write_last(self):
    # Append a zero-filled final step.
    final_step = utils.final_step_like(self._buffer[0], self._next_observation)
    self._writer.append(final_step)

    # The length of the sequence we will be adding is the size of the buffer
    # plus one due to the final step.
    steps = list(self._buffer) + [final_step]
    num_steps = len(steps)

    # Calculate the priority for this episode.
    table_priorities = utils.calculate_priorities(self._priority_fns, steps)

    # Create a prioritized item for each table.
    for table_name, priority in table_priorities.items():
      self._writer.create_item(table_name, num_steps, priority)
