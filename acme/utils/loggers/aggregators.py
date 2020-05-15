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

"""Utilities for aggregating to other loggers."""

from typing import Callable, Optional, Sequence
from acme.utils.loggers import base


class Dispatcher(base.Logger):
  """Writes data to multiple `Logger` objects."""

  def __init__(
      self, to: Sequence[base.Logger],
      serialize_fn: Optional[Callable[[base.LoggingData], str]] = None):
    """Initialize `Dispatcher` connected to several `Logger` objects."""
    self._to = to
    self._serialize_fn = serialize_fn

  def write(self, values: base.LoggingData):
    """Writes `values` to the underlying `Logger` objects."""
    if self._serialize_fn:
      values = self._serialize_fn(values)
    for logger in self._to:
      logger.write(values)
