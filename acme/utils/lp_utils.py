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

"""Utility function for building and launching launchpad programs."""

import functools
import inspect
import time
from typing import Any, Callable

from absl import flags
from absl import logging
from acme.utils import counting
import launchpad as lp

FLAGS = flags.FLAGS


def partial_kwargs(function: Callable[..., Any],
                   **kwargs: Any) -> Callable[..., Any]:
  """Return a partial function application by overriding default keywords.

  This function is equivalent to `functools.partial(function, **kwargs)` but
  will raise a `ValueError` when called if either the given keyword arguments
  are not defined by `function` or if they do not have defaults.

  This is useful as a way to define a factory function with default parameters
  and then to override them in a safe way.

  Args:
    function: the base function before partial application.
    **kwargs: keyword argument overrides.

  Returns:
    A function.
  """
  # Try to get the argspec of our function which we'll use to get which keywords
  # have defaults.
  argspec = inspect.getfullargspec(function)

  # Figure out which keywords have defaults.
  if argspec.defaults is None:
    defaults = []
  else:
    defaults = argspec.args[-len(argspec.defaults):]

  # Find any keys not given as defaults by the function.
  unknown_kwargs = set(kwargs.keys()).difference(defaults)

  # Raise an error
  if unknown_kwargs:
    error_string = 'Cannot override unknown or non-default kwargs: {}'
    raise ValueError(error_string.format(', '.join(unknown_kwargs)))

  return functools.partial(function, **kwargs)


class StepsLimiter:
  """Process that terminates an experiment when `max_steps` is reached."""

  def __init__(self,
               counter: counting.Counter,
               max_steps: int,
               steps_key: str = 'actor_steps'):
    self._counter = counter
    self._max_steps = max_steps
    self._steps_key = steps_key

  def run(self):
    """Run steps limiter to terminate an experiment when max_steps is reached.
    """

    logging.info('StepsLimiter: Starting with max_steps = %d (%s)',
                 self._max_steps, self._steps_key)
    while True:
      # Update the counts.
      counts = self._counter.get_counts()
      num_steps = counts.get(self._steps_key, 0)

      logging.info('StepsLimiter: Reached %d recorded steps', num_steps)

      if num_steps > self._max_steps:
        logging.info('StepsLimiter: Max steps of %d was reached, terminating',
                     self._max_steps)
        lp.stop()

      # Don't spam the counter.
      time.sleep(10.)
