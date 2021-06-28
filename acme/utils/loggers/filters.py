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

"""Loggers which filter other loggers."""

import math
import time
from typing import Callable, Optional, Sequence

from acme.utils.loggers import base


class NoneFilter(base.Logger):
  """Logger which writes to another logger, filtering any `None` values."""

  def __init__(self, to: base.Logger):
    """Initializes the logger.

    Args:
      to: A `Logger` object to which the current object will forward its results
        when `write` is called.
    """
    self._to = to

  def write(self, values: base.LoggingData):
    values = {k: v for k, v in values.items() if v is not None}
    self._to.write(values)

  def close(self):
    self._to.close()


class TimeFilter(base.Logger):
  """Logger which writes to another logger at a given time interval."""

  def __init__(self, to: base.Logger, time_delta: float):
    """Initializes the logger.

    Args:
      to: A `Logger` object to which the current object will forward its results
        when `write` is called.
      time_delta: How often to write values out in seconds.
        Note that writes within `time_delta` are dropped.
    """
    self._to = to
    self._time = 0
    self._time_delta = time_delta
    if time_delta < 0:
      raise ValueError(f'time_delta must be greater than 0 (got {time_delta}).')

  def write(self, values: base.LoggingData):
    now = time.time()
    if (now - self._time) > self._time_delta:
      self._to.write(values)
      self._time = now

  def close(self):
    self._to.close()


class KeyFilter(base.Logger):
  """Logger which filters keys in logged data."""

  def __init__(
      self,
      to: base.Logger,
      *,
      keep: Optional[Sequence[str]] = None,
      drop: Optional[Sequence[str]] = None,
  ):
    """Creates the filter.

    Args:
      to: A `Logger` object to which the current object will forward its writes.
      keep: Keys that are kept by the filter. Note that `keep` and `drop` cannot
        be both set at once.
      drop: Keys that are dropped by the filter. Note that `keep` and `drop`
        cannot be both set at once.
    """
    if bool(keep) == bool(drop):
      raise ValueError('Exactly one of `keep` & `drop` arguments must be set.')
    self._to = to
    self._keep = keep
    self._drop = drop

  def write(self, data: base.LoggingData):
    if self._keep:
      data = {k: data[k] for k in self._keep}
    if self._drop:
      data = {k: v for k, v in data.items() if k not in self._drop}
    self._to.write(data)

  def close(self):
    self._to.close()


class GatedFilter(base.Logger):
  """Logger which writes to another logger based on a gating function.

  This logger tracks the number of times its `write` method is called, and uses
  a gating function on this number to decide when to write.
  """

  def __init__(self, to: base.Logger, gating_fn: Callable[[int], bool]):
    """Initialises the logger.

    Args:
      to: A `Logger` object to which the current object will forward its results
        when `write` is called.
      gating_fn: A function that takes an integer (number of calls) as input.
        For example, to log every tenth call: gating_fn=lambda t: t % 10 == 0.
    """
    self._to = to
    self._gating_fn = gating_fn
    self._calls = 0

  def write(self, values: base.LoggingData):
    if self._gating_fn(self._calls):
      self._to.write(values)
    self._calls += 1

  def close(self):
    self._to.close()

  @classmethod
  def logarithmic(cls, to: base.Logger, n: int = 10) -> 'GatedFilter':
    """Builds a logger for writing at logarithmically-spaced intervals.

    This will log on a linear scale at each order of magnitude of `n`.
    For example, with n=10, this will log at times:
        [0, 1, 2, ..., 9, 10, 20, 30, ... 90, 100, 200, 300, ... 900, 1000]

    Args:
      to: The underlying logger to write to.
      n: Base (default 10) on which to operate.
    Returns:
      A GatedFilter logger, which gates logarithmically as described above.
    """
    def logarithmic_filter(t: int) -> bool:
      magnitude = math.floor(math.log10(max(t, 1))/math.log10(n))
      return t % (n**magnitude) == 0
    return cls(to, gating_fn=logarithmic_filter)

  @classmethod
  def periodic(cls, to: base.Logger, interval: int = 10) -> 'GatedFilter':
    """Builds a logger for writing at linearly-spaced intervals.

    Args:
      to: The underlying logger to write to.
      interval: The interval between writes.
    Returns:
      A GatedFilter logger, which gates periodically as described above.
    """
    return cls(to, gating_fn=lambda t: t % interval == 0)
