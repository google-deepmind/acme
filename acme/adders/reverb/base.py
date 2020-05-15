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

"""Adders that use Reverb (github.com/deepmind/reverb) as a backend."""

import abc
import collections
from typing import Callable, Mapping, NamedTuple, Optional

from acme import types
from acme.adders import base

import dm_env
import reverb

DEFAULT_PRIORITY_TABLE = 'priority_table'


class Step(NamedTuple):
  """Step class used internally for reverb adders."""
  observation: types.NestedArray
  action: types.NestedArray
  reward: types.NestedArray
  discount: types.NestedArray
  extras: types.NestedArray


class PriorityFnInput(NamedTuple):
  """The input to a priority function consisting of stacked steps."""
  observations: types.NestedArray
  actions: types.NestedArray
  rewards: types.NestedArray
  discounts: types.NestedArray
  extras: types.NestedArray


# Define the type of a priority function and the mapping from table to function.
PriorityFn = Callable[['PriorityFnInput'], float]
PriorityFnMapping = Mapping[str, PriorityFn]


class ReverbAdder(base.Adder):
  """Base class for Reverb adders."""

  def __init__(
      self,
      client: reverb.Client,
      buffer_size: int,
      max_sequence_length: int,
      delta_encoded: bool = False,
      chunk_length: Optional[int] = None,
      priority_fns: Optional[PriorityFnMapping] = None,
  ):
    """Initialize a ReverbAdder instance.

    Args:
      client: A client to the Reverb backend.
      buffer_size: Number of steps to retain in memory.
      max_sequence_length: The maximum length of sequences (corresponding to the
        number of observations) that can be added to replay.
      delta_encoded: If `True` (False by default) enables delta encoding, see
        `Client` for more information.
      chunk_length: Number of timesteps grouped together before delta encoding
        and compression. See `Client` for more information.
      priority_fns: A mapping from table names to priority functions; if
        omitted, all transitions/steps/sequences are given uniform priorities
        (1.0) and placed in DEFAULT_PRIORITY_TABLE.
    """
    if priority_fns:
      priority_fns = dict(priority_fns)
    else:
      priority_fns = {DEFAULT_PRIORITY_TABLE: lambda x: 1.}

    self._client = client
    self._priority_fns = priority_fns
    self._max_sequence_length = max_sequence_length
    self._delta_encoded = delta_encoded
    self._chunk_length = chunk_length

    # This is exposed as the _writer property in such a way that it will create
    # a new writer automatically whenever the internal __writer is None. Users
    # should ONLY ever interact with self._writer.
    self.__writer = None

    # The state of the adder is captured by a buffer of `buffer_size` steps
    # (generally SAR tuples) and one additional dangling observation.
    self._buffer = collections.deque(maxlen=buffer_size)
    self._next_observation = None

  @property
  def _writer(self) -> reverb.Writer:
    if self.__writer is None:
      self.__writer = self._client.writer(
          self._max_sequence_length,
          delta_encoded=self._delta_encoded,
          chunk_length=self._chunk_length)
    return self.__writer

  def add_priority_table(self, table_name: str, priority_fn: PriorityFn):
    if table_name in self._priority_fns:
      raise ValueError(
          'A priority function already exists for {}.'.format(table_name))
    self._priority_fns[table_name] = priority_fn

  def reset(self):
    """Resets the adder's buffer."""
    if self.__writer:
      self._writer.close()
      self.__writer = None
    self._buffer.clear()
    self._next_observation = None

  def add_first(self, timestep: dm_env.TimeStep):
    """Record the first observation of a trajectory."""
    if not timestep.first():
      raise ValueError('adder.add_first with an initial timestep (i.e. one for '
                       'which timestep.first() is True')

    if self._next_observation is not None:
      raise ValueError('adder.reset must be called before adder.add_first '
                       '(called automatically if `next_timestep.last()` is '
                       'true when `add` is called).')

    # Record the next observation.
    self._next_observation = timestep.observation

  def add(self,
          action: types.NestedArray,
          next_timestep: dm_env.TimeStep,
          extras: types.NestedArray = ()):
    """Record an action and the following timestep."""
    if self._next_observation is None:
      raise ValueError('adder.add_first must be called before adder.add.')

    # Add the timestep to the buffer.
    self._buffer.append(
        Step(
            observation=self._next_observation,
            action=action,
            reward=next_timestep.reward,
            discount=next_timestep.discount,
            extras=extras,
        ))

    # Record the next observation and write.
    self._next_observation = next_timestep.observation
    self._write()

    # Write the last "dangling" observation.
    if next_timestep.last():
      self._write_last()
      self.reset()

  @abc.abstractmethod
  def _write(self):
    """Write data to replay from the buffer."""

  @abc.abstractmethod
  def _write_last(self):
    """Write data to replay from the buffer."""
