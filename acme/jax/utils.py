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

"""Utilities for JAX."""

import queue
import threading
from typing import Iterable, Generator, TypeVar

from absl import logging
from acme import types
import haiku as hk
import jax
from jax import tree_util
import jax.numpy as jnp
import numpy as np
import tree


def add_batch_dim(values: types.Nest) -> types.NestedArray:
  return tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), values)


@hk.transform
def _flatten(x, num_batch_dims: int):
  return hk.Flatten(preserve_dims=num_batch_dims)(x)


def batch_concat(
    values: types.NestedArray,
    num_batch_dims: int = 1,
) -> jnp.ndarray:
  """Flatten and concatenate nested array structure, keeping batch dims."""
  flatten_fn = lambda x: _flatten.apply(None, x, num_batch_dims)
  flat_leaves = tree.map_structure(flatten_fn, values)
  return jnp.concatenate(tree.flatten(flat_leaves), axis=-1)


def zeros_like(nest: types.Nest) -> types.NestedArray:
  return tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), nest)


def squeeze_batch_dim(nest: types.Nest) -> types.NestedArray:
  return tree_util.tree_map(lambda x: jnp.squeeze(x, axis=0), nest)


def to_numpy_squeeze(values: types.Nest) -> types.NestedArray:
  """Converts to numpy and squeezes out dummy batch dimension."""
  return tree_util.tree_map(lambda x: np.array(x).squeeze(axis=0), values)


def fetch_devicearray(values: types.Nest) -> types.Nest:
  """Fetches and converts any DeviceArrays in `values`."""

  def _serialize(x):
    if isinstance(x, jnp.DeviceArray):
      return np.array(x)
    return x
  return tree.map_structure(_serialize, values)


def batch_to_sequence(values: types.Nest) -> types.NestedArray:
  return tree_util.tree_map(
      lambda x: jnp.transpose(x, axes=(1, 0, *range(2, len(x.shape)))), values)


T = TypeVar('T')


def prefetch(iterable: Iterable[T],
             buffer_size: int = 5,
             device=None) -> Generator[T, None, None]:
  """Performs prefetching of elements from an iterable in a separate thread.

  Args:
    iterable: A python iterable. This is used to build the python prefetcher.
      Note that each iterable should only be passed to this function once as
      iterables aren't thread safe
    buffer_size (int): Number of elements to keep in the prefetch buffer.
    device: The device to prefetch the elements to. If none then the elements
      are left on the CPU. The device should be of the type returned by
      `jax.devices()`.

  Yields:
    Prefetched elements from the original iterable.
  Raises:
    ValueError if the buffer_size <= 1.
    Any error thrown by the iterable_function. Note this is not raised inside
      the producer, but after it finishes executing.
  """

  if buffer_size <= 1:
    raise ValueError('the buffer_size should be > 1')
  buffer = queue.Queue(maxsize=(buffer_size - 1))
  producer_error = []
  end = object()

  def producer():
    """Enqueues items from `iterable` on a given thread."""
    try:
      # Build a new iterable for each thread. This is crucial if working with
      # tensorflow datasets because tf.Graph objects are thread local.
      for item in iterable:
        if device:
          item = jax.device_put(item, device)
        buffer.put(item)
    except Exception as e:  # pylint: disable=broad-except
      logging.exception('Error in producer thread for %s', iterable.__name__)
      producer_error.append(e)
    finally:
      buffer.put(end)

  # Start the producer thread.
  threading.Thread(target=producer, daemon=True).start()

  # Consume from the buffer.
  while True:
    value = buffer.get()
    if value is end:
      break
    yield value

  if producer_error:
    raise producer_error[0]
