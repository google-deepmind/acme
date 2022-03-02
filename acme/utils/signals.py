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

"""Helper methods for handling signals."""

import contextlib
import ctypes
import threading
from typing import Any, Callable, Optional

import launchpad

_Handler = Callable[[], Any]


@contextlib.contextmanager
def runtime_terminator(callback: Optional[_Handler] = None):
  """Runtime terminator used for stopping computation upon agent termination.

    Runtime terminator optionally executed a provided `callback` and then raises
    `SystemExit` exception in the thread performing the computation.

  Args:
    callback: callback to execute before raising exception.

  Yields:
      None.
  """
  worker_id = threading.get_ident()
  def signal_handler():
    if callback:
      callback()
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(worker_id), ctypes.py_object(SystemExit))
    assert res < 2, 'Stopping worker failed'
  launchpad.register_stop_handler(signal_handler)
  yield
  launchpad.unregister_stop_handler(signal_handler)
