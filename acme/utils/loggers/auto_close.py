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

"""Logger which self closes on exit if not closed yet."""

import weakref

from acme.utils.loggers import base


class AutoCloseLogger(base.Logger):
  """Logger which auto closes itself on exit if not already closed."""

  def __init__(self, logger: base.Logger):
    self._logger = logger
    # The finalizer "logger.close" is invoked in one of the following scenario:
    # 1) the current logger is GC
    # 2) from the python doc, when the program exits, each remaining live
    #    finalizer is called.
    # Note that in the normal flow, where "close" is explicitly called,
    # the finalizer is marked as dead using the detach function so that
    # the underlying logger is not closed twice (once explicitly and once
    # implicitly when the object is GC or when the program exits).
    self._finalizer = weakref.finalize(self, logger.close)

  def write(self, values: base.LoggingData):
    self._logger.write(values)

  def close(self):
    if self._finalizer.detach():
      self._logger.close()
    self._logger = None
