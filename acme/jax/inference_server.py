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

"""Defines Inference Server class used for centralised inference."""

import dataclasses
import datetime
import threading
from typing import Any, Callable, Mapping, Optional, Sequence, Union
import acme
from acme.jax import variable_utils
import jax
import launchpad as lp


@dataclasses.dataclass
class InferenceServerConfig:
  """Configuration options for centralised inference.

  Attributes:
    batch_size: How many elements to batch together per single inference call.
        Auto-computed when not specified.
    update_period: Frequency of updating variables from the variable source.
        It is passed to VariableClient. Auto-computed when not specified.
    timeout: Time after which incomplete batch is executed (batch is padded,
        so there batch handler is always called with batch_size elements).
        By default timeout is effectively disabled (set to 30 days).
  """
  batch_size: Optional[int] = None
  update_period: Optional[int] = None
  timeout: datetime.timedelta = datetime.timedelta(days=30)

CallableOrMapping = Union[Callable[..., Any], Mapping[str, Callable[..., Any]]]


class InferenceServer:
  """Centralised, batched inference server."""

  def __init__(self,
               handler: CallableOrMapping,
               variable_source: acme.VariableSource,
               devices: Sequence[jax.xla.Device],
               config: InferenceServerConfig):
    """Constructs an inference server object.

    Args:
      handler: A callable or a mapping of callables to be exposed
        through the inference server.
      variable_source: Source of variables
      devices: Devices used for executing handlers. All devices are used in
        parallel.
      config: Inference Server configuration.
    """
    self._variable_source = variable_source
    self._variable_client = None
    self._keys = []
    self._devices = devices
    self._config = config
    self._call_cnt = 0
    self._device_params = [None] * len(self._devices)
    self._device_params_ids = [None] * len(self._devices)
    self._mutex = threading.Lock()

    if callable(handler):
      self._build_handler(handler, '__call__')
    else:
      for name in handler:
        self._build_handler(handler[name], name)

  def _dereference_params(self, arg):
    """Replaces VariableReferences with their corresponding param values."""

    if not isinstance(arg, variable_utils.VariableReference):
      # All arguments but VariableReference are returned without modifications.
      return arg

    # Due to batching dimension we take the first element.
    variable_name = arg.variable_name[0]

    if variable_name not in self._keys:
      # Create a new VariableClient which also serves new variables.
      self._keys.append(variable_name)
      self._variable_client = variable_utils.VariableClient(
          client=self._variable_source,
          key=self._keys,
          update_period=self._config.update_period)
    self._variable_client.update()
    # Maybe update params, depending on client configuration.
    params = self._variable_client.params
    device_idx = self._call_cnt % len(self._devices)
    # Select device via round robin, and update its params if they changed.
    if self._device_params_ids[device_idx] != id(params):
      self._device_params_ids[device_idx] = id(params)
      self._device_params[device_idx] = jax.device_put(
          params, self._devices[device_idx])

    if len(self._keys) == 1:
      return params
    return params[self._keys.index(variable_name)]

  def _build_handler(self, handler: Callable[..., Any], name: str):
    """Builds a batched handler for a given callable handler and its name."""

    def dereference_params_and_call_handler(*args, **kwargs):
      with self._mutex:
        # Dereference args corresponding to params, leaving others unchanged.
        args_with_dereferenced_params = map(self._dereference_params, args)
        self._call_cnt += 1
      return handler(*args_with_dereferenced_params, **kwargs)

    self.__setattr__(
        name,
        lp.batched_handler(
            batch_size=self._config.batch_size,
            timeout=self._config.timeout,
            pad_batch=True,
            max_parallelism=2 *
            len(self._devices))(dereference_params_and_call_handler))
