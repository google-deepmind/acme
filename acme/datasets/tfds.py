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
from typing import Any, Dict, Iterator, Tuple

from acme import types
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds


def _episode_to_transition(step: Dict[str, Any]) -> types.Transition:
  return types.Transition(
      observation=step['observation'][:-1],
      action=step['action'][:-1],
      reward=step['reward'][:-1],
      discount=1.0 - tf.cast(step['is_terminal'][1:], dtype=tf.float32),
      # If next step is terminal, then the observation may be arbitrary.
      next_observation=step['observation'][1:],
  )


def _episode_steps_to_transition(episode) -> tf.data.Dataset:
  """Transforms an Episode into a dataset of Transitions."""
  episode = episode['steps']
  # The code below might fail if the dataset contains more than 1e9 transitions,
  # which is quite unlikely.
  data = tf.data.experimental.get_single_element(episode.batch(1000000000))
  data = _episode_to_transition(data)
  return tf.data.Dataset.from_tensor_slices(data)


def get_tfds_dataset(dataset_name: str):
  dataset = tfds.load(dataset_name)
  return dataset['train'].flat_map(_episode_steps_to_transition)


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
               batch_size: int):
    """Creates an iterator.

    Args:
      dataset: underlying tf Dataset
      key: a key to be used for random number generation
      batch_size: batch size
    """
    # Read the whole dataset. We use artificially large batch_size to make sure
    # we capture the whole dataset.
    data = next(dataset.batch(1000000000).as_numpy_iterator())
    self._dataset_size = jax.tree_flatten(
        jax.tree_map(lambda x: x.shape[0], data))[0][0]
    self._jax_dataset = jax.tree_map(jnp.asarray, data)
    logging.info('Finished loading a dataset into memory. Elements: %d',
                 self._dataset_size)
    self._key = key

    def sample(key: jnp.ndarray) -> Tuple[Any, jnp.ndarray]:
      key, key_randint = jax.random.split(key)
      indices = jax.random.randint(key_randint, (batch_size,), minval=0,
                                   maxval=self._dataset_size)
      demo_transitions = jax.tree_map(lambda d: jnp.take(d, indices, axis=0),
                                      self._jax_dataset)
      return demo_transitions, key
    self._sample = jax.jit(sample)

  def __next__(self) -> Any:
    data, self._key = self._sample(self._key)
    return data

  @property
  def dataset_size(self):
    """An integer of the dataset cardinality."""
    return self._dataset_size
