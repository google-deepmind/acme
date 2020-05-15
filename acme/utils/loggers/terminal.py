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

"""Utilities for logging to the terminal."""

import time
from typing import Any, Callable

from acme.utils.loggers import base
import numpy as np


def _format_key(key: str) -> str:
  """Internal function for formatting keys."""
  return key.replace('_', ' ').title()


def _format_value(value: Any) -> str:
  """Internal function for formatting values."""
  value = base.to_numpy(value)
  if isinstance(value, (float, np.number)):
    return f'{value:0.3f}'
  return f'{value}'


def serialize(values: base.LoggingData) -> str:
  """Converts `values` to a pretty-printed string.

  This takes a dictionary `values` whose keys are strings and returns a
  a formatted string such that each key, value pair is separated by ' = ' and
  each entry is separated by ' | '. The keys are sorted alphabetically to ensure
  a consistent order, and snake case is split into words.

  For example:

      values = {'a': 1, 'b' = 2.33333333, 'c': 'hello', 'big_value': 10}
      # Returns 'A = 1 | B = 2.333 | Big Value = 10 | C = hello'
      values_string = serialize(values)

  Args:
    values: A dictionary with string keys.

  Returns:
    A formatted string.
  """
  return ' | '.join(f'{_format_key(k)} = {_format_value(v)}'
                    for k, v in sorted(values.items()))


class TerminalLogger(base.Logger):
  """Logs to terminal."""

  def __init__(
      self,
      label: str = '',
      print_fn: Callable[[str], None] = print,
      serialize_fn: Callable[[base.LoggingData], str] = serialize,
      time_delta: float = 0.0,
  ):
    """Initializes the logger.

    Args:
      label: label string to use when logging.
      print_fn: function to call which acts like print.
      serialize_fn: function to call which transforms values into a str.
      time_delta: How often (in seconds) to write values. This can be used to
        minimize terminal spam, but is 0 by default---ie everything is written.
    """

    self._print_fn = print_fn
    self._serialize_fn = serialize_fn
    self._label = label and f'[{_format_key(label)}] '
    self._time = time.time()
    self._time_delta = time_delta

  def write(self, values: base.LoggingData):
    now = time.time()
    if (now - self._time) > self._time_delta:
      self._print_fn(f'{self._label}{self._serialize_fn(values)}')
      self._time = now
