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

"""Loggers which filter other loggers."""

import time

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


class TimeFilter(base.Logger):
  """Logger which writes to another logger at a given time interval."""

  def __init__(self, to: base.Logger, time_delta: float):
    """Initializes the logger.

    Args:
      to: A `Logger` object to which the current object will forward its results
        when `write` is called.
      time_delta: How often to write values out in seconds.
    """
    self._to = to
    self._time = time.time()
    self._time_delta = time_delta

  def write(self, values: base.LoggingData):
    now = time.time()
    if (now - self._time) > self._time_delta:
      self._to.write(values)
      self._time = now
