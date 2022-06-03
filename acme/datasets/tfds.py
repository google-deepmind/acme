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

"""Utilities related to loading TFDS datasets."""

import logging
from typing import Any, Iterator, Optional, Tuple, Sequence

from acme import specs
from acme import types
from flax import jax_utils
import jax
import jax.numpy as jnp
import numpy as np
import rlds
import tensorflow as tf
import tensorflow_datasets as tfds


def _batched_step_to_transition(step: rlds.BatchedStep) -> types.Transition:
  return types.Transition(
      observation=tf.nest.map_structure(lambda x: x[0], step[rlds.OBSERVATION]),
      action=tf.nest.map_structure(lambda x: x[0], step[rlds.ACTION]),
      reward=tf.nest.map_structure(lambda x: x[0], step[rlds.REWARD]),
      discount=1.0 - tf.cast(step[rlds.IS_TERMINAL][1], dtype=tf.float32),
      # If next step is terminal, then the observation may be arbitrary.
      next_observation=tf.nest.map_structure(
          lambda x: x[1], step[rlds.OBSERVATION])
  )


def _batch_steps(episode: rlds.Episode) -> tf.data.Dataset:
  return rlds.transformations.batch(
      episode[rlds.STEPS], size=2, shift=1, drop_remainder=True)


def _dataset_size_upperbound(dataset: tf.data.Dataset) -> int:
  if dataset.cardinality() != tf.data.experimental.UNKNOWN_CARDINALITY:
    return dataset.cardinality()
  return tf.cast(
      dataset.batch(1000).reduce(0, lambda x, step: x + 1000), tf.int64)


def load_tfds_dataset(
    dataset_name: str,
    num_episodes: Optional[int] = None,
    env_spec: Optional[specs.EnvironmentSpec] = None) -> tf.data.Dataset:
  """Returns a TFDS dataset with the given name."""
  # Used only in tests.
  del env_spec

  dataset = tfds.load(dataset_name)['train']
  if num_episodes:
    dataset = dataset.take(num_episodes)
  return dataset


# TODO(sinopalnikov): replace get_ftds_dataset with a pair of load/transform.
def get_tfds_dataset(
    dataset_name: str,
    num_episodes: Optional[int] = None,
    env_spec: Optional[specs.EnvironmentSpec] = None) -> tf.data.Dataset:
  """Returns a TFDS dataset transformed to a dataset of transitions."""
  dataset = load_tfds_dataset(dataset_name, num_episodes, env_spec)
  batched_steps = dataset.flat_map(_batch_steps)
  return rlds.transformations.map_steps(batched_steps,
                                        _batched_step_to_transition)


# In order to avoid excessive copying on TPU one needs to make the last
# dimension a multiple of this number.
_BEST_DIVISOR = 128


def _pad(x: jnp.ndarray) -> jnp.ndarray:
  if len(x.shape) != 2:
    return x
  # Find a more scientific way to find this threshold (30). Depending on various
  # conditions for low enough sizes the excessive copying is not triggered.
  if x.shape[-1] % _BEST_DIVISOR != 0 and x.shape[-1] > 30:
    n = _BEST_DIVISOR - (x.shape[-1] % _BEST_DIVISOR)
    x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, n)], 'constant')
  return x


# Undo the padding.
def _unpad(x: jnp.ndarray, shape: Sequence[int]) -> jnp.ndarray:
  if len(shape) == 2 and x.shape[-1] != shape[-1]:
    return x[..., :shape[-1]]
  return x


_PMAP_AXIS_NAME = 'data'


class JaxInMemoryRandomSampleIterator(Iterator[Any]):
  """In memory random sample iterator implemented in JAX.

  Loads the whole dataset in memory and performs random sampling with
  replacement of batches of `batch_size`.
  This class provides much faster sampling functionality compared to using
  an iterator on tf.data.Dataset.
  """

  def __init__(self,
               dataset: tf.data.Dataset,
               key: jnp.ndarray,
               batch_size: int,
               shard_dataset_across_devices: bool = False):
    """Creates an iterator.

    Args:
      dataset: underlying tf Dataset
      key: a key to be used for random number generation
      batch_size: batch size
      shard_dataset_across_devices: whether to use all available devices
        for storing the underlying dataset. The upside is a larger
        dataset capacity that fits into memory. Downsides are:
          - execution of pmapped functions is usually slower than jitted
          - few last elements in the dataset might be dropped (if not multiple)
          - sampling is not 100% uniform, since each core will be doing sampling
            only within its data chunk
        The number of available devices must divide the batch_size evenly.
    """
    # Read the whole dataset. We use artificially large batch_size to make sure
    # we capture the whole dataset.
    size = _dataset_size_upperbound(dataset)
    data = next(dataset.batch(size).as_numpy_iterator())
    self._dataset_size = jax.tree_flatten(
        jax.tree_map(lambda x: x.shape[0], data))[0][0]
    device = jax_utils._pmap_device_order()
    if not shard_dataset_across_devices:
      device = device[:1]
    should_pmap = len(device) > 1
    assert batch_size % len(device) == 0
    self._dataset_size = self._dataset_size - self._dataset_size % len(device)
    # len(device) needs to divide self._dataset_size evenly.
    assert self._dataset_size % len(device) == 0
    logging.info('Trying to load %s elements to %s', self._dataset_size, device)
    logging.info('Dataset %s %s',
                 ('before padding' if should_pmap else ''),
                 jax.tree_map(lambda x: x.shape, data))
    if should_pmap:
      shapes = jax.tree_map(lambda x: x.shape, data)
      # Padding to a multiple of 128 is needed to avoid excessive copying on TPU
      data = jax.tree_map(_pad, data)
      logging.info('Dataset after padding %s',
                   jax.tree_map(lambda x: x.shape, data))
      def split_and_put(x: jnp.ndarray) -> jnp.ndarray:
        return jax.device_put_sharded(
            np.split(x[:self._dataset_size], len(device)), devices=device)
      self._jax_dataset = jax.tree_map(split_and_put, data)
    else:
      self._jax_dataset = jax.tree_map(jax.device_put, data)

    self._key = (jnp.stack(jax.random.split(key, len(device)))
                 if should_pmap else key)

    def sample_per_shard(data: Any,
                         key: jnp.ndarray) -> Tuple[Any, jnp.ndarray]:
      key1, key2 = jax.random.split(key)
      indices = jax.random.randint(
          key1, (batch_size // len(device),),
          minval=0,
          maxval=self._dataset_size // len(device))
      data_sample = jax.tree_map(lambda d: jnp.take(d, indices, axis=0), data)
      return data_sample, key2

    if should_pmap:
      def sample(data, key):
        data_sample, key = sample_per_shard(data, key)
        # Gathering data on TPUs is much more efficient that doing so on a host
        # since it avoids Host - Device communications.
        data_sample = jax.lax.all_gather(
            data_sample, axis_name=_PMAP_AXIS_NAME, axis=0, tiled=True)
        data_sample = jax.tree_multimap(_unpad, data_sample, shapes)
        return data_sample, key

      pmapped_sample = jax.pmap(sample, axis_name=_PMAP_AXIS_NAME)

      def sample_and_postprocess(key: jnp.ndarray) -> Tuple[Any, jnp.ndarray]:
        data, key = pmapped_sample(self._jax_dataset, key)
        # All pmapped devices return the same data, so we just take the one from
        # the first device.
        return jax.tree_map(lambda x: x[0], data), key
      self._sample = sample_and_postprocess
    else:
      self._sample = jax.jit(
          lambda key: sample_per_shard(self._jax_dataset, key))

  def __next__(self) -> Any:
    data, self._key = self._sample(self._key)
    return data

  @property
  def dataset_size(self) -> int:
    """An integer of the dataset cardinality."""
    return self._dataset_size
