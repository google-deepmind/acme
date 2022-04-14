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

"""Utilities to use within loggers."""

import queue
import threading
from typing import Callable, TypeVar, Generic

from absl import logging


E = TypeVar("E")


class AsyncExecutor(Generic[E]):
  """Executes a blocking function asynchronously on a queue of items."""

  def __init__(
      self,
      fn: Callable[[E], None],
      queue_size: int = 1,
      interruptible_interval_secs: float = 1.0,
  ):
    """Buffers elements in a queue and runs `fn` asynchronously..

    NOTE: Once closed, `AsyncExecutor` will block until current `fn` finishes
      but is not guaranteed to dequeue all elements currently stored in
      the data queue. This is intentional so as to prevent a blocking `fn` call
      from preventing `AsyncExecutor` from closing.

    Args:
      fn: A callable to be executed upon dequeuing an element from data
        queue.
      queue_size: The maximum size of the synchronized buffer queue.
      interruptible_interval_secs: Timeout interval in seconds for blocking
        queue operations after which the background threads check for errors and
        if background threads should stop.
    """
    self._data = queue.Queue(maxsize=queue_size)
    self._should_stop = threading.Event()
    self._errors = queue.Queue()
    self._interruptible_interval_secs = interruptible_interval_secs

    def _dequeue() -> None:
      """Dequeue data from a queue and invoke blocking call."""
      while not self._should_stop.is_set():
        try:
          element = self._data.get(timeout=self._interruptible_interval_secs)
          # Execute fn upon dequeuing an element from the data queue.
          fn(element)
        except queue.Empty:
          # If queue is Empty for longer than the specified time interval,
          # check again if should_stop has been requested and retry.
          continue
        except Exception as e:
          logging.error("AsyncExecuter thread terminated with error.")
          logging.exception(e)
          self._errors.put(e)
          self._should_stop.set()
          raise  # Never caught by anything, just terminates the thread.

    self._thread = threading.Thread(target=_dequeue, daemon=True)
    self._thread.start()

  def _raise_on_error(self) -> None:
    try:
      # Raise the error on the caller thread if an error has been raised in the
      # looper thread.
      raise self._errors.get_nowait()
    except queue.Empty:
      pass

  def close(self):
    self._should_stop.set()
    # Join all background threads.
    self._thread.join()
    # Raise errors produced by background threads.
    self._raise_on_error()

  def put(self, element: E) -> None:
    """Puts `element` asynchronuously onto the underlying data queue.

    The write call blocks if the underlying data_queue contains `queue_size`
      elements for over `self._interruptible_interval_secs` second, in which
      case we check if stop has been requested or if there has been an error
      raised on the looper thread. If neither happened, retry enqueue.

    Args:
      element: an element to be put into the underlying data queue and dequeued
        asynchronuously for `fn(element)` call.
    """
    while not self._should_stop.is_set():
      try:
        self._data.put(element, timeout=self._interruptible_interval_secs)
        break
      except queue.Full:
        continue
    else:
      # If `should_stop` has been set, then raises if any has been raised on
      # the background thread.
      self._raise_on_error()
