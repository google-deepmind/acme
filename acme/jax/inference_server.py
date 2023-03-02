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
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar
import acme
from acme.jax import variable_utils
import jax
import launchpad as lp



import numpy as np

from jax import numpy as jnp
# from jax.lib import pytree
from jax.tree_util import tree_flatten, tree_unflatten

def tree_stack(trees):
  """Takes a list of trees and stacks every corresponding leaf.
  For example, given two trees ((a, b), c) and ((a', b'), c'), returns
  ((stack(a, a'), stack(b, b')), stack(c, c')).
  Useful for turning a list of objects into something you can feed to a
  vmapped function.
  """
  leaves_list = []
  treedef_list = []
  for tree in trees:
    leaves, treedef = tree_flatten(tree)
    leaves_list.append(leaves)
    treedef_list.append(treedef)

  grouped_leaves = zip(*leaves_list)
  result_leaves = [jnp.stack(l) for l in grouped_leaves]
  # return treedef_list[0].unflatten(result_leaves), treedef_list[0]
  return treedef_list[0].unflatten(result_leaves)

def tree_unstack(tree):
  """Takes a tree and turns it into a list of trees. Inverse of tree_stack.
  For example, given a tree ((a, b), c), where a, b, and c all have first
  dimension k, will make k trees
  [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
  Useful for turning the output of a vmapped function into normal objects.
  """
  leaves, treedef = tree_flatten(tree)
  n_trees = leaves[0].shape[0]
  new_leaves = [[] for _ in range(n_trees)]
  for leaf in leaves:
    for i in range(n_trees):
      new_leaves[i].append(leaf[i])
  new_trees = [treedef.unflatten(l) for l in new_leaves]
  return new_trees

tree_stack = jax.jit(tree_stack)
tree_unstack = jax.jit(tree_unstack)

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


InferenceServerHandler = TypeVar('InferenceServerHandler')



def reverso_oar_thing(oar_thing):
  from acme.wrappers.observation_action_reward import OAR
  num_things = len(oar_thing.action)
  new_oars = [OAR(
    action=oar_thing.action[i],
    reward=oar_thing.reward[i],
    observation=oar_thing.observation[i]
    ) for i in range(num_things)]
  return new_oars
  

class InferenceServer(Generic[InferenceServerHandler]):
  """Centralised, batched inference server."""

  def __init__(self, handler: InferenceServerHandler,
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
    self._handler = jax.tree_map(self._build_handler, handler, is_leaf=callable)

  @property
  def handler(self) -> InferenceServerHandler:
    return self._handler

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

    params = self._variable_client.params
    device_idx = self._call_cnt % len(self._devices)
    # Select device via round robin, and update its params if they changed.
    if self._device_params_ids[device_idx] != id(params):
      self._device_params_ids[device_idx] = id(params)
      self._device_params[device_idx] = jax.device_put(
          params, self._devices[device_idx])

    # Return the params that are located on the chosen device.
    device_params = self._device_params[device_idx]
    if len(self._keys) == 1:
      return device_params
    return device_params[self._keys.index(variable_name)]

  def _build_handler(self, handler: Callable[..., Any]) -> Callable[..., Any]:
    """Builds a batched handler for a given callable handler and its name."""

    def dereference_params_and_call_handler(*args, **kwargs):
      with self._mutex:
        # Dereference args corresponding to params, leaving others unchanged.
        args_with_dereferenced_params = [
            self._dereference_params(arg) for arg in args
        ]
        kwargs_with_dereferenced_params = {
            key: self._dereference_params(value)
            for key, value in kwargs.items()
        }
        self._call_cnt += 1

        # Maybe update params, depending on client configuration.
        if self._variable_client is not None:
          self._variable_client.update()
      params = args_with_dereferenced_params[0]
      oar_thing = args_with_dereferenced_params[1]
      reversed_oar = reverso_oar_thing(oar_thing)
      recurrent_state_thing = args_with_dereferenced_params[2]
      reversed_oar_stacked = tree_stack(reversed_oar)
      recurrent_state_thing_stacked = tree_stack(recurrent_state_thing)
      to_return = handler(params, reversed_oar_stacked, recurrent_state_thing_stacked)
      unstacked_to_return = tree_unstack(to_return)
      return unstacked_to_return
      # return handler(*args_with_dereferenced_params,
      #                **kwargs_with_dereferenced_params)

    to_return = lp.batched_handler(
        batch_size=self._config.batch_size,
        timeout=self._config.timeout,
        pad_batch=True,
        max_parallelism=2 * len(self._devices))(
            dereference_params_and_call_handler)
    return to_return
