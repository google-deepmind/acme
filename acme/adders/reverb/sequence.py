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

"""Sequence adders.

This implements adders which add sequences or partial trajectories.
"""

from typing import Optional

from acme.adders.reverb import base
from acme.adders.reverb import utils

import numpy as np
import reverb
import tree


class SequenceAdder(base.ReverbAdder):
  """An which adds sequences of fixed length."""

  def __init__(
      self,
      client: reverb.Client,
      sequence_length: int,
      period: int,
      delta_encoded: bool = False,
      chunk_length: Optional[int] = None,
      priority_fns: Optional[base.PriorityFnMapping] = None,
  ):
    """Makes a SequenceAdder instance.

    Args:
      client: See docstring for BaseAdder.
      sequence_length: The fixed length of sequences we wish to add.
      period: The period with which we add sequences. If less than
        sequence_length, overlapping sequences are added. If equal to
        sequence_length, sequences are exactly non-overlapping.
      delta_encoded: If `True` (False by default) enables delta encoding, see
        `Client` for more information.
      chunk_length: Number of timesteps grouped together before delta encoding
        and compression. See `Client` for more information.
      priority_fns: See docstring for BaseAdder.
    """
    super().__init__(
        client=client,
        buffer_size=sequence_length,
        max_sequence_length=sequence_length,
        delta_encoded=delta_encoded,
        chunk_length=chunk_length,
        priority_fns=priority_fns)

    self._period = period
    self._step = 0

  def reset(self):
    self._step = 0
    super().reset()

  def _write(self):
    # Append the previous step and increment number of steps written.
    self._writer.append(self._buffer[-1])
    self._step += 1
    self._maybe_add_priorities()

  def _write_last(self):
    # Create a final step and a step full of zeros.
    final_step = utils.final_step_like(self._buffer[0], self._next_observation)
    zero_step = tree.map_structure(np.zeros_like, final_step)

    # Append the final step.
    self._buffer.append(final_step)
    self._writer.append(final_step)
    self._step += 1

    # NOTE: this always pads to the fixed length. but this is not equivalent to
    # the old Padded sequence adder.

    # Determine how much padding to add. This makes sure that we add (zero) data
    # until the next time we would write a sequence.
    if self._step <= self._max_sequence_length:
      padding = self._max_sequence_length - self._step
    else:
      padding = self._period - (self._step - self._max_sequence_length)

    # Pad with zeros to get a full sequence.
    for _ in range(padding):
      self._buffer.append(zero_step)
      self._writer.append(zero_step)
      self._step += 1

    # Write priorities for the sequence.
    self._maybe_add_priorities()

  def _maybe_add_priorities(self):
    if not (
        # Write the first time we hit the max sequence length...
        self._step == self._max_sequence_length or
        # ... or every `period`th time after hitting max length.
        (self._step > self._max_sequence_length and
         (self._step - self._max_sequence_length) % self._period == 0)):
      return

    # Compute priorities for the buffer.
    steps = list(self._buffer)
    num_steps = len(steps)
    table_priorities = utils.calculate_priorities(self._priority_fns, steps)

    # Create a prioritized item for each table.
    for table_name, priority in table_priorities.items():
      self._writer.create_item(table_name, num_steps, priority)
