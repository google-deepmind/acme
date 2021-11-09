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

"""A thin wrapper around Python's builtin signal.signal()."""
import signal
import types
from typing import Any, Callable

_Handler = Callable[[], Any]


def add_handler(signo: signal.Signals, fn: _Handler):

  # The function signal.signal expects the handler to take an int rather than a
  # signal.Signals.
  def _wrapped(signo: int, frame: types.FrameType):
    del signo, frame
    return fn()

  signal.signal(signo.value, _wrapped)
