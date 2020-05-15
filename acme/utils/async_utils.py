# Lint as: python3
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

from concurrent import futures
import sys

from absl import logging
import wrapt


def make_async(thread_name_prefix=''):
  """Returns a decorator that runs any function it wraps in a background thread.

   When called, the decorated function will immediately return a future
   representing its result.
   The function being decorated can be an instance method or normal function.
   Consecutive calls to the decorated function are guaranteed to be in order
   and non overlapping.
   An error raised by the decorated function will be raised in the background
   thread at call-time. Raising the error in the main thread is deferred until
   the next call, so as to be non-blocking.
   All subsequent calls to the decorated function after an error has been raised
   will not run (regardless of whether the arguments have changed); instead
   they will re-raise the original error in the main thread.

  Args:
    thread_name_prefix: Str prefix for the background thread, for easier
    debugging.

  Returns:
    decorator that runs any function it wraps in a background thread, and
    handles any errors raised.
  """
  # We have a single thread pool per wrapped function to ensure that calls to
  # the function are run in order (but in a background thread).
  pool = futures.ThreadPoolExecutor(max_workers=1,
                                    thread_name_prefix=thread_name_prefix)
  errors = []
  @wrapt.decorator
  def decorator(wrapped, instance, args, kwargs):
    """Runs wrapped in a background thread so result is non-blocking.

    Args:
      wrapped: A function to wrap and execute in background thread.
        Can be instance method or normal function.
      instance: The object to which the wrapped function was bound when it was
        called (None if wrapped is a normal function).
      args: List of position arguments supplied when wrapped function
        was called.
      kwargs: Dict of keyword arguments supplied when the wrapped function was
        called.

    Returns:
      A future representing the result of calling wrapped.
    Raises:
      Exception object caught in background thread, if call to wrapped fails.
      Exception object with stacktrace in main thread, if the previous call to
        wrapped failed.
    """
    del instance

    def trap_errors(*args, **kwargs):
      """Wraps wrapped to trap any errors thrown."""

      if errors:
        # Do not execute wrapped if previous call errored.
        return
      try:
        return wrapped(*args, **kwargs)
      except Exception as e:
        errors.append(sys.exc_info())
        logging.exception(
            'Error in producer thread for {%s}', thread_name_prefix)
        raise e

    if errors:
      # Previous call had an error, re-raise in main thread.
      exc_info = errors[-1]
      raise exc_info[1].with_traceback(exc_info[2])
    return pool.submit(trap_errors, *args, **kwargs)
  return decorator
