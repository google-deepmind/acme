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
import datetime
import time
from typing import List, NamedTuple, Optional, Sequence, Union

from acme import core
from acme.jax import networks as network_types
import jax


class VariableReference(NamedTuple):
  variable_name: str


class ReferenceVariableSource(core.VariableSource):
  """Variable source which returns references instead of values.

  This is passed to each actor when using a centralized inference server. The
  actor uses this special variable source to get references rather than values.
  These references are then passed to calls to the inference server, which will
  dereference them to obtain the value of the corresponding variables at
  inference time. This avoids passing around copies of variables from each
  actor to the inference server.
  """

  def get_variables(self, names: List[str]) -> List[VariableReference]:
    return [VariableReference(name) for name in names]


class VariableClient:
  """A variable client for updating variables from a remote source."""

  def __init__(self,
               client: core.VariableSource,
               key: Union[str, Sequence[str]],
               update_period: Union[int, datetime.timedelta] = 1,
               device: Optional[Union[str, jax.xla.Device]] = None):
    """Initializes the variable client.

    Args:
      client: A variable source from which we fetch variables.
      key: Which variables to request. When multiple keys are used, params
        property will return a list of params.
      update_period: Interval between fetches, specified as either (int) a
        number of calls to update() between actual fetches or (timedelta) a time
        interval that has to pass since the last fetch.
      device: The name of a JAX device to put variables on. If None (default),
        VariableClient won't put params on any device.
    """
    self._update_period = update_period
    self._call_counter = 0
    self._last_call = time.time()
    self._client = client
    self._params: Sequence[network_types.Params] = None

    self._device = device
    if isinstance(self._device, str):
      self._device = jax.devices(device)[0]

    self._executor = futures.ThreadPoolExecutor(max_workers=1)

    if isinstance(key, str):
      key = [key]

    self._key = key
    self._request = lambda k=key: client.get_variables(k)
    self._future: Optional[futures.Future] = None  # pylint: disable=g-bare-generic
    self._async_request = lambda: self._executor.submit(self._request)

  def update(self, wait: bool = False) -> None:
    """Periodically updates the variables with the latest copy from the source.

    If wait is True, a blocking request is executed. Any active request will be
    cancelled.
    If wait is False, this method makes an asynchronous request for variables.

    Args:
      wait: Whether to execute asynchronous (False) or blocking updates (True).
        Defaults to False.
    """
    # Track calls (we only update periodically).
    self._call_counter += 1

    # Return if it's not time to fetch another update.
    if isinstance(self._update_period, datetime.timedelta):
      if self._update_period.total_seconds() + self._last_call > time.time():
        return
    else:
      if self._call_counter < self._update_period:
        return

    if wait:
      if self._future is not None:
        if self._future.running():
          self._future.cancel()
        self._future = None
      self._call_counter = 0
      self._last_call = time.time()
      self.update_and_wait()
      return

    # Return early if we are still waiting for a previous request to come back.
    if self._future and not self._future.done():
      return

    # Get a future and add the copy function as a callback.
    self._call_counter = 0
    self._last_call = time.time()
    self._future = self._async_request()
    self._future.add_done_callback(lambda f: self._callback(f.result()))

  def update_and_wait(self):
    """Immediately update and block until we get the result."""
    self._callback(self._request())

  def _callback(self, params_list: List[network_types.Params]):
    if self._device and not isinstance(self._client, ReferenceVariableSource):
      # Move variables to a proper device.
      self._params = jax.device_put(params_list, self._device)
    else:
      self._params = params_list

  @property
  def device(self) -> Optional[jax.xla.Device]:
    return self._device

  @property
  def params(self) -> Union[network_types.Params, List[network_types.Params]]:
    """Returns the first params for one key, otherwise the whole params list."""
    if self._params is None:
      self.update_and_wait()

    if len(self._params) == 1:
      return self._params[0]
    else:
      return self._params
