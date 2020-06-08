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

"""Variable utilities for JAX."""

from concurrent import futures
from typing import List

from acme import core

import haiku as hk


class VariableClient:
  """A variable client for updating variables from a remote source."""

  def __init__(self,
               client: core.VariableSource,
               key: str,
               update_period: int = 1):
    self._key = key
    self._update_period = update_period
    self._call_counter = 0
    self._client = client
    self._params = None

    self._executor = futures.ThreadPoolExecutor(max_workers=1)
    self._request = lambda: client.get_variables([self._key])
    self._future = futures.Future()
    self._async_request = lambda: self._executor.submit(self._request)

  def update(self):
    """Periodically updates the variables with latest copy from the source."""

    # Track calls (we only update periodically).
    if self._call_counter < self._update_period:
      self._call_counter += 1

    # Return early if we are still waiting for a previous request to come back.
    if self._future.running():
      return

    # Return if it's not time to fetch another update.
    if self._call_counter < self._update_period:
      return

    # Get a future and add the copy function as a callback.
    self._call_counter = 0
    self._future = self._async_request()
    self._future.add_done_callback(lambda f: self._callback(f.result()))

  def update_and_wait(self):
    """Immediately update and block until we get the result."""
    self._callback(self._request())

  def _callback(self, params_list: List[hk.Params]):
    assert len(params_list) == 1
    self._params = params_list[0]

  @property
  def params(self) -> hk.Params:
    if self._params is None:
      self.update_and_wait()
    return self._params
