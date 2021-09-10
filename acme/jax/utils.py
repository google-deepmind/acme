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

import functools
import itertools
import queue
import threading
from typing import Callable, Generator, Iterable, NamedTuple, Optional, Sequence, Tuple, TypeVar

from absl import logging
from acme import types
from acme.jax.types import TrainingMetrics, TrainingStepOutput  # pylint: disable=g-multiple-import
import jax
import jax.numpy as jnp
import numpy as np
import tree

F = TypeVar('F', bound=Callable)
N = TypeVar('N', bound=types.NestedArray)
T = TypeVar('T')


def add_batch_dim(values: types.Nest) -> types.NestedArray:
  return jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), values)


def _flatten(x: jnp.ndarray, num_batch_dims: int) -> jnp.ndarray:
  """Flattens the input, preserving the first ``num_batch_dims`` dimensions.

  If the input has fewer than ``num_batch_dims`` dimensions, it is returned
  unchanged.
  If the input has exactly ``num_batch_dims`` dimensions, an extra dimension
  is added. This is needed to handle batched scalars.

  Arguments:
    x: the input array to flatten.
    num_batch_dims: number of dimensions to preserve.

  Returns:
    flattened input.
  """
  # TODO(b/173492429): consider throwing an error instead.
  if x.ndim < num_batch_dims:
    return x
  return jnp.reshape(x, list(x.shape[:num_batch_dims]) + [-1])


def batch_concat(
    values: types.NestedArray,
    num_batch_dims: int = 1,
) -> jnp.ndarray:
  """Flatten and concatenate nested array structure, keeping batch dims."""
  flatten_fn = lambda x: _flatten(x, num_batch_dims)
  flat_leaves = tree.map_structure(flatten_fn, values)
  return jnp.concatenate(tree.flatten(flat_leaves), axis=-1)


def zeros_like(nest: types.Nest, dtype=None) -> types.NestedArray:
  return jax.tree_map(lambda x: jnp.zeros(x.shape, dtype or x.dtype), nest)


def ones_like(nest: types.Nest, dtype=None) -> types.NestedArray:
  return jax.tree_map(lambda x: jnp.ones(x.shape, dtype or x.dtype), nest)


def squeeze_batch_dim(nest: types.Nest) -> types.NestedArray:
  return jax.tree_map(lambda x: jnp.squeeze(x, axis=0), nest)


def to_numpy_squeeze(values: types.Nest) -> types.NestedArray:
  """Converts to numpy and squeezes out dummy batch dimension."""
  return jax.tree_map(lambda x: np.asarray(x).squeeze(axis=0), values)


def to_numpy(values: types.Nest) -> types.NestedArray:
  return jax.tree_map(np.asarray, values)


def fetch_devicearray(values: types.Nest) -> types.Nest:
  """Fetches and converts any DeviceArrays to np.ndarrays."""
  return tree.map_structure(_fetch_devicearray, values)


def _fetch_devicearray(x):
  if isinstance(x, jax.xla.DeviceArray):
    return np.asarray(x)
  return x


def batch_to_sequence(values: types.Nest) -> types.NestedArray:
  return jax.tree_map(
      lambda x: jnp.transpose(x, axes=(1, 0, *range(2, len(x.shape)))), values)


def tile_array(array: jnp.ndarray, multiple: int) -> jnp.ndarray:
  """Tiles `multiple` copies of `array` along a new leading axis."""
  return jnp.stack([array] * multiple)


def tile_nested(inputs: types.Nest, multiple: int) -> types.Nest:
  """Tiles tensors in a nested structure along a new leading axis."""
  tile = functools.partial(tile_array, multiple=multiple)
  return jax.tree_map(tile, inputs)


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
      logging.exception('Error in producer thread for %s', iterable)
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


class PrefetchingSplit(NamedTuple):
  host: types.NestedArray
  device: types.NestedArray


_SplitFunction = Callable[[types.NestedArray], PrefetchingSplit]


def sharded_prefetch(
    iterable: Iterable[types.NestedArray],
    buffer_size: int = 5,
    num_threads: int = 1,
    split_fn: Optional[_SplitFunction] = None,
    devices: Optional[Sequence[jax.xla.Device]] = None,
) -> Generator[types.NestedArray, None, None]:
  """Performs sharded prefetching from an iterable in separate threads.

  Elements from the resulting generator are intended to be used in a jax.pmap
  call. Every element is a sharded prefetched array with an additional replica
  dimension and corresponds to jax.local_device_count() elements from the
  original iterable.

  Args:
    iterable: A python iterable. This is used to build the python prefetcher.
      Note that each iterable should only be passed to this function once as
      iterables aren't thread safe.
    buffer_size (int): Number of elements to keep in the prefetch buffer.
    num_threads (int): Number of threads.
    split_fn: Optional function applied to every element from the iterable to
      split the parts of it that will be kept in the host and the parts that
      will sent to the device.
    devices: Devices used for prefecthing. Optional, jax.local_devices() by
      default.

  Yields:
    Prefetched elements from the original iterable with additional replica
    dimension.
  Raises:
    ValueError if the buffer_size <= 1.
    Any error thrown by the iterable_function. Note this is not raised inside
      the producer, but after it finishes executing.
  """

  devices = devices or jax.local_devices()

  if buffer_size <= 1:
    raise ValueError('the buffer_size should be > 1')
  buffer = queue.Queue(maxsize=(buffer_size - 1))
  producer_error = []
  end = object()

  def producer():
    """Enqueues batched items from `iterable` on a given thread."""
    try:
      # Build a new iterable for each thread. This is crucial if working with
      # tensorflow datasets because tf.Graph objects are thread local.
      it = iter(iterable)
      while True:
        items = itertools.islice(it, len(devices))
        if not items:
          break
        if split_fn is None:
          buffer.put(jax.device_put_sharded(tuple(items), devices))
        else:
          # ((host: x1, device: y1), ..., (host: xN, device: yN)).
          items_split = (split_fn(item) for item in items)
          # (host: (x1, ..., xN), device: (y1, ..., yN)).
          split = tree.map_structure_up_to(
              PrefetchingSplit(None, None), lambda *x: x, *items_split)

          buffer.put(
              PrefetchingSplit(
                  host=np.stack(split.host),
                  device=jax.device_put_sharded(split.device, devices)))
    except Exception as e:  # pylint: disable=broad-except
      logging.exception('Error in producer thread for %s', iterable)
      producer_error.append(e)
    finally:
      buffer.put(end)

  # Start producer threads.
  for _ in range(num_threads):
    threading.Thread(target=producer, daemon=True).start()

  # Consume from the buffer.
  while True:
    value = buffer.get()
    if value is end:
      break
    yield value

  if producer_error:
    raise producer_error[0]


def replicate_in_all_devices(nest: N,
                             devices: Optional[Sequence[jax.xla.Device]] = None
                            ) -> N:
  """Replicate array nest in all available devices."""
  devices = devices or jax.local_devices()
  return jax.device_put_sharded([nest] * len(devices), devices)


def get_from_first_device(nest: N, as_numpy: bool = True) -> N:
  """Gets the first array of a nest of `jax.pxla.ShardedDeviceArray`s.

  Args:
    nest: A nest of `jax.pxla.ShardedDeviceArray`s.
    as_numpy: If `True` then each `DeviceArray` that is retrieved is transformed
      (and copied if not on the host machine) into a `np.ndarray`.

  Returns:
    The first array of a nest of `jax.pxla.ShardedDeviceArray`s. Note that if
    `as_numpy=False` then the array will be a `DeviceArray` (which will live on
    the same device as the sharded device array). If `as_numpy=True` then the
    array will be copied to the host machine and converted into a `np.ndarray`.
  """

  def _check_type_and_slice(x):
    if not isinstance(x, jax.pxla.ShardedDeviceArray):
      raise ValueError('get_from_first_device should only be used with '
                       f'{jax.pxla.ShardedDeviceArray}, passed {type(x)}.')
    return x[0]

  zeroth_nest = jax.tree_map(_check_type_and_slice, nest)

  return jax.device_get(zeroth_nest) if as_numpy else zeroth_nest


def mapreduce(
    f: F,
    reduce_fn: Optional[Callable[[jnp.DeviceArray], jnp.DeviceArray]] = None,
    **vmap_kwargs,
) -> F:
  """A simple decorator that transforms `f` into (`reduce_fn` o vmap o f).

  By default, we vmap over axis 0, and the `reduce_fn` is jnp.mean over axis 0.
  Note that the call signature of `f` is invariant under this transformation.

  If, for example, f has shape signature [H, W] -> [N], then mapreduce(f)
  (with the default arguments) will have shape signature [B, H, W] -> [N].

  Args:
    f: A pure function over examples.
    reduce_fn: A pure function that reduces DeviceArrays -> DeviceArrays.
    **vmap_kwargs: Keyword arguments to forward to `jax.vmap`.

  Returns:
    g: A pure function over batches of examples.
  """

  if reduce_fn is None:
    reduce_fn = lambda x: jnp.mean(x, axis=0)

  vmapped_f = jax.vmap(f, **vmap_kwargs)

  def g(*args, **kwargs):
    return jax.tree_map(reduce_fn, vmapped_f(*args, **kwargs))

  return g


_TrainingState = TypeVar('_TrainingState')
_TrainingData = TypeVar('_TrainingData')
_TrainingAux = TypeVar('_TrainingAux')


# TODO(b/192806089): migrate all callers to process_many_batches and remove this
# method.
def process_multiple_batches(
    process_one_batch: Callable[[_TrainingState, _TrainingData],
                                Tuple[_TrainingState, _TrainingAux]],
    num_batches: int,
    postprocess_aux: Optional[Callable[[_TrainingAux], _TrainingAux]] = None
) -> Callable[[_TrainingState, _TrainingData], Tuple[_TrainingState,
                                                     _TrainingAux]]:
  """Makes 'process_one_batch' process multiple batches at once.

  Args:
    process_one_batch: a function that takes 'state' and 'data', and returns
      'new_state' and 'aux' (for example 'metrics').
    num_batches: how many batches to process at once
    postprocess_aux: how to merge the extra information, defaults to taking
      the mean.
  Returns:
    A function with the same interface as 'process_one_batch' which processes
    multiple batches at once.
  """
  assert num_batches >= 1
  if num_batches == 1:
    if not postprocess_aux:
      return process_one_batch
    def _process_one_batch(state, data):
      state, aux = process_one_batch(state, data)
      return state, postprocess_aux(aux)
    return _process_one_batch

  if postprocess_aux is None:
    postprocess_aux = lambda x: jax.tree_map(jnp.mean, x)

  def _process_multiple_batches(state, data):
    data = jax.tree_map(
        lambda a: jnp.reshape(a, (num_batches, -1, *a.shape[1:])), data)

    state, aux = jax.lax.scan(
        process_one_batch, state, data, length=num_batches)
    return state, postprocess_aux(aux)

  return _process_multiple_batches


def process_many_batches(
    process_one_batch: Callable[[_TrainingState, _TrainingData],
                                TrainingStepOutput[_TrainingState]],
    num_batches: int,
    postprocess_aux: Optional[Callable[[TrainingMetrics],
                                       TrainingMetrics]] = None
) -> Callable[[_TrainingState, _TrainingData],
              TrainingStepOutput[_TrainingState]]:
  """The version of 'process_multiple_batches' with stronger typing."""

  def _process_one_batch(
      state: _TrainingState,
      data: _TrainingData) -> Tuple[_TrainingState, TrainingMetrics]:
    result = process_one_batch(state, data)
    return result.state, result.metrics

  func = process_multiple_batches(_process_one_batch, num_batches,
                                  postprocess_aux)

  def _process_many_batches(
      state: _TrainingState,
      data: _TrainingData) -> TrainingStepOutput[_TrainingState]:
    state, aux = func(state, data)
    return TrainingStepOutput(state, aux)

  return _process_many_batches


def weighted_softmax(x: jnp.ndarray, weights: jnp.ndarray, axis: int = 0):
  x = x - jnp.max(x, axis=axis)
  return weights * jnp.exp(x) / jnp.sum(weights * jnp.exp(x),
                                        axis=axis, keepdims=True)
