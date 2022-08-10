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
from typing import Callable, Iterable, Iterator, NamedTuple, Optional, Sequence, Tuple, TypeVar

from absl import logging
from acme import core
from acme import types
from acme.jax import types as jax_types
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tree


F = TypeVar('F', bound=Callable)
N = TypeVar('N', bound=types.NestedArray)
T = TypeVar('T')


NUM_PREFETCH_THREADS = 1


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


def maybe_recover_lstm_type(state: types.NestedArray) -> types.NestedArray:
  """Recovers the type hk.LSTMState if LSTMState is in the type name.

  When the recurrent state of recurrent neural networks (RNN) is deserialized,
  for example when it is sampled from replay, it is sometimes repacked in a type
  that is identical to the source type but not the correct type itself. When
  using this state as the initial state in an hk.dynamic_unroll, this will
  cause hk.dynamic_unroll to raise an error as it requires its input and output
  states to be identical.

  Args:
    state: a nested structure of arrays representing the state of an RNN.

  Returns:
    Either the state unchanged if it is anything but an LSTMState, otherwise
    returns the state arrays properly contained in an hk.LSTMState.
  """
  return hk.LSTMState(*state) if type(state).__name__ == 'LSTMState' else state


def prefetch(
    iterable: Iterable[T],
    buffer_size: int = 5,
    device: Optional[jax.xla.Device] = None,
    num_threads: int = NUM_PREFETCH_THREADS,
) -> core.PrefetchingIterator[T]:
  """Returns prefetching iterator with additional 'ready' method."""

  return PrefetchIterator(iterable, buffer_size, device, num_threads)


class PrefetchingSplit(NamedTuple):
  host: types.NestedArray
  device: types.NestedArray


_SplitFunction = Callable[[types.NestedArray], PrefetchingSplit]


def device_put(
    iterable: Iterable[types.NestedArray],
    device: jax.xla.Device,
    split_fn: Optional[_SplitFunction] = None,
):
  """Returns iterator that samples an item and places it on the device."""

  return PutToDevicesIterable(
      iterable=iterable,
      pmapped_user=False,
      devices=[device],
      split_fn=split_fn)


def multi_device_put(
    iterable: Iterable[types.NestedArray],
    devices: Sequence[jax.xla.Device],
    split_fn: Optional[_SplitFunction] = None,
):
  """Returns iterator that, per device, samples an item and places on device."""

  return PutToDevicesIterable(
      iterable=iterable, pmapped_user=True, devices=devices, split_fn=split_fn)


class PutToDevicesIterable(Iterable[types.NestedArray]):
  """Per device, samples an item from iterator and places on device.

  if pmapped_user:
    Items from the resulting generator are intended to be used in a pmapped
    function. Every element is a ShardedDeviceArray or (nested) Python container
    thereof. A single next() call to this iterator results in len(devices)
    calls to the underlying iterator. The returned items are put one on each
    device.
  if not pmapped_user:
    Places a sample from the iterator on the given device.

  Yields:
    If no split_fn is specified:
      DeviceArray/ShardedDeviceArray or (nested) Python container thereof
      representing the elements of shards stacked together, with each shard
      backed by physical device memory specified by the corresponding entry in
      devices.

    If split_fn is specified:
      PrefetchingSplit where the .host element is a stacked numpy array or
      (nested) Python contained thereof. The .device element is a
      DeviceArray/ShardedDeviceArray or (nested) Python container thereof.

  Raises:
    StopIteration: if there are not enough items left in the iterator to place
      one sample on each device.
    Any error thrown by the iterable_function. Note this is not raised inside
      the producer, but after it finishes executing.
  """

  def __init__(
      self,
      iterable: Iterable[types.NestedArray],
      pmapped_user: bool,
      devices: Sequence[jax.xla.Device],
      split_fn: Optional[_SplitFunction] = None,
  ):
    """Constructs PutToDevicesIterable.

    Args:
      iterable: A python iterable. This is used to build the python prefetcher.
        Note that each iterable should only be passed to this function once as
        iterables aren't thread safe.
      pmapped_user: whether the user of data from this iterator is implemented
        using pmapping.
      devices: Devices used for prefecthing.
      split_fn: Optional function applied to every element from the iterable to
        split the parts of it that will be kept in the host and the parts that
        will sent to the device.

    Raises:
      ValueError: If devices list is empty, or if pmapped_use=False and more
        than 1 device is provided.
    """
    self.num_devices = len(devices)
    if self.num_devices == 0:
      raise ValueError('At least one device must be specified.')
    if (not pmapped_user) and (self.num_devices != 1):
      raise ValueError('User is not implemented with pmapping but len(devices) '
                       f'= {len(devices)} is not equal to 1! Devices given are:'
                       f'\n{devices}')

    self.iterable = iterable
    self.pmapped_user = pmapped_user
    self.split_fn = split_fn
    self.devices = devices
    self.iterator = iter(self.iterable)

  def __iter__(self) -> Iterator[types.NestedArray]:
    # It is important to structure the Iterable like this, because in
    # JustPrefetchIterator we must build a new iterable for each thread.
    # This is crucial if working with tensorflow datasets because tf.Graph
    # objects are thread local.
    self.iterator = iter(self.iterable)
    return self

  def __next__(self) -> types.NestedArray:
    try:
      if not self.pmapped_user:
        item = next(self.iterator)
        if self.split_fn is None:
          return jax.device_put(item, self.devices[0])
        item_split = self.split_fn(item)
        return PrefetchingSplit(
            host=item_split.host,
            device=jax.device_put(item_split.device, self.devices[0]))

      items = itertools.islice(self.iterator, self.num_devices)
      items = tuple(items)
      if len(items) < self.num_devices:
        raise StopIteration
      if self.split_fn is None:
        return jax.device_put_sharded(tuple(items), self.devices)
      else:
        # ((host: x1, device: y1), ..., (host: xN, device: yN)).
        items_split = (self.split_fn(item) for item in items)
        # (host: (x1, ..., xN), device: (y1, ..., yN)).
        split = tree.map_structure_up_to(
            PrefetchingSplit(None, None), lambda *x: x, *items_split)

        return PrefetchingSplit(
            host=np.stack(split.host),
            device=jax.device_put_sharded(split.device, self.devices))

    except StopIteration:
      raise

    except Exception:  # pylint: disable=broad-except
      logging.exception('Error for %s', self.iterable)
      raise


def sharded_prefetch(
    iterable: Iterable[types.NestedArray],
    buffer_size: int = 5,
    num_threads: int = 1,
    split_fn: Optional[_SplitFunction] = None,
    devices: Optional[Sequence[jax.xla.Device]] = None,
) -> core.PrefetchingIterator:
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

  Returns:
    Prefetched elements from the original iterable with additional replica
    dimension.
  Raises:
    ValueError if the buffer_size <= 1.
    Any error thrown by the iterable_function. Note this is not raised inside
      the producer, but after it finishes executing.
  """

  devices = devices or jax.local_devices()

  iterable = PutToDevicesIterable(
      iterable=iterable, pmapped_user=True, devices=devices, split_fn=split_fn)

  return prefetch(iterable, buffer_size, device=None, num_threads=num_threads)


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
  zeroth_nest = jax.tree_map(lambda x: x[0], nest)
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
    postprocess_aux: how to merge the extra information, defaults to taking the
      mean.

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
                                jax_types.TrainingStepOutput[_TrainingState]],
    num_batches: int,
    postprocess_aux: Optional[Callable[[jax_types.TrainingMetrics],
                                       jax_types.TrainingMetrics]] = None
) -> Callable[[_TrainingState, _TrainingData],
              jax_types.TrainingStepOutput[_TrainingState]]:
  """The version of 'process_multiple_batches' with stronger typing."""

  def _process_one_batch(
      state: _TrainingState,
      data: _TrainingData) -> Tuple[_TrainingState, jax_types.TrainingMetrics]:
    result = process_one_batch(state, data)
    return result.state, result.metrics

  func = process_multiple_batches(_process_one_batch, num_batches,
                                  postprocess_aux)

  def _process_many_batches(
      state: _TrainingState,
      data: _TrainingData) -> jax_types.TrainingStepOutput[_TrainingState]:
    state, aux = func(state, data)
    return jax_types.TrainingStepOutput(state, aux)

  return _process_many_batches


def weighted_softmax(x: jnp.ndarray, weights: jnp.ndarray, axis: int = 0):
  x = x - jnp.max(x, axis=axis)
  return weights * jnp.exp(x) / jnp.sum(weights * jnp.exp(x),
                                        axis=axis, keepdims=True)


def sample_uint32(random_key: jax_types.PRNGKey) -> int:
  """Returns an integer uniformly distributed in 0..2^32-1."""
  iinfo = jnp.iinfo(jnp.int32)
  # randint only accepts int32 values as min and max.
  jax_random = jax.random.randint(
      random_key, shape=(), minval=iinfo.min, maxval=iinfo.max, dtype=jnp.int32)
  return np.uint32(jax_random).item()


class PrefetchIterator(core.PrefetchingIterator):
  """Performs prefetching from an iterable in separate threads.

  Its interface is additionally extended with `ready` method which tells whether
  there is any data waiting for processing and a `retrieved_elements` method
  specifying number of elements retrieved from the iterator.

  Yields:
    Prefetched elements from the original iterable.

  Raises:
    ValueError: if the buffer_size < 1.
    StopIteration: If the iterable contains no more items.
    Any error thrown by the iterable_function. Note this is not raised inside
      the producer, but after it finishes executing.
  """

  def __init__(
      self,
      iterable: Iterable[types.NestedArray],
      buffer_size: int = 5,
      device: Optional[jax.xla.Device] = None,
      num_threads: int = NUM_PREFETCH_THREADS,
  ):
    """Constructs PrefetchIterator.

    Args:
      iterable: A python iterable. This is used to build the python prefetcher.
        Note that each iterable should only be passed to this function once as
        iterables aren't thread safe.
      buffer_size (int): Number of elements to keep in the prefetch buffer.
      device (deprecated): Optionally place items from the iterable on the given
        device. If None, the items are returns as given by the iterable. This
        argument is deprecated and the recommended usage is to wrap the
        iterables using utils.device_put or utils.multi_device_put before using
        utils.prefetch.
      num_threads (int): Number of threads.
    """

    if buffer_size < 1:
      raise ValueError('the buffer_size should be >= 1')
    self.buffer = queue.Queue(maxsize=buffer_size)
    self.producer_error = []
    self.end = object()
    self.iterable = iterable
    self.device = device
    self.count = 0

    # Start producer threads.
    for _ in range(num_threads):
      threading.Thread(target=self.producer, daemon=True).start()

  def producer(self):
    """Enqueues items from `iterable` on a given thread."""
    try:
      # Build a new iterable for each thread. This is crucial if working with
      # tensorflow datasets because tf.Graph objects are thread local.
      for item in self.iterable:
        if self.device:
          jax.device_put(item, self.device)
        self.buffer.put(item)
    except Exception as e:  # pylint: disable=broad-except
      logging.exception('Error in producer thread for %s', self.iterable)
      self.producer_error.append(e)
    finally:
      self.buffer.put(self.end)

  def __iter__(self):
    return self

  def ready(self):
    return not self.buffer.empty()

  def retrieved_elements(self):
    return self.count

  def __next__(self):
    value = self.buffer.get()
    if value is self.end:
      if self.producer_error:
        raise self.producer_error[0] from self.producer_error[0]
      raise StopIteration
    self.count += 1
    return value
