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

"""Variable handling utilities for TensorFlow 2."""

from concurrent import futures
from typing import Mapping, Optional, Sequence

from acme import core

import tensorflow as tf
import tree


class VariableClient:
  """A variable client for updating variables from a remote source."""

  def __init__(self,
               client: core.VariableSource,
               variables: Mapping[str, Sequence[tf.Variable]],
               update_period: int = 1):
    self._keys = list(variables.keys())
    self._variables = tree.flatten(list(variables.values()))
    self._call_counter = 0
    self._update_period = update_period
    self._client = client
    self._request = lambda: client.get_variables(self._keys)

    # Create a single background thread to fetch variables without necessarily
    # blocking the actor.
    self._executor = futures.ThreadPoolExecutor(max_workers=1)
    self._async_request = lambda: self._executor.submit(self._request)

    # Initialize this client's future to None to indicate to the `update()`
    # method that there is no pending/running request.
    self._future: Optional[futures.Future] = None

  def update(self, wait: bool = False):
    """Periodically updates the variables with the latest copy from the source.

    This stateful update method keeps track of the number of calls to it and,
    every `update_period` call, sends a request to its server to retrieve the
    latest variables.

    If wait is True, a blocking request is executed. Any active request will be
    cancelled.
    If wait is False, this method makes an asynchronous request for variables
    and returns. Unless the request is immediately fulfilled, the variables are
    only copied _within a subsequent call to_ `update()`, whenever the request
    is fulfilled by the `VariableSource`. If there is an existing fulfilled
    request when this method is called, the resulting variables are immediately
    copied.

    Args:
      wait: if True, executes blocking update.
    """
    # Track the number of calls (we only update periodically).
    if self._call_counter < self._update_period:
      self._call_counter += 1

    period_reached: bool = self._call_counter >= self._update_period

    if period_reached and wait:
      # Cancel any active request.
      self._future: Optional[futures.Future] = None
      self.update_and_wait()
      self._call_counter = 0
      return

    if period_reached and self._future is None:
      # The update period has been reached and no request has been sent yet, so
      # making an asynchronous request now.
      self._future = self._async_request()
      self._call_counter = 0

    if self._future is not None and self._future.done():
      # The active request is done so copy the result and remove the future.
      self._copy(self._future.result())
      self._future: Optional[futures.Future] = None
    else:
      # There is either a pending/running request or we're between update
      # periods, so just carry on.
      return

  def update_and_wait(self):
    """Immediately update and block until we get the result."""
    self._copy(self._request())

  def _copy(self, new_variables: Sequence[Sequence[tf.Variable]]):
    """Copies the new variables to the old ones."""

    new_variables = tree.flatten(new_variables)
    if len(self._variables) != len(new_variables):
      raise ValueError('Length mismatch between old variables and new.')

    for new, old in zip(new_variables, self._variables):
      old.assign(new)
