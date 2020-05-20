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

"""Functions for making TensorFlow datasets for sampling from Reverb replay."""

from typing import Optional

from acme import specs
from acme import types
from acme.adders import reverb as adders

import reverb
import tensorflow as tf
import tree


def make_reverb_dataset(
    client: reverb.TFClient,
    environment_spec: specs.EnvironmentSpec,
    batch_size: Optional[int] = None,
    prefetch_size: Optional[int] = None,
    sequence_length: Optional[int] = None,
    extra_spec: Optional[types.NestedSpec] = None,
    transition_adder: bool = False,
    table: str = adders.DEFAULT_PRIORITY_TABLE,
    parallel_batch_optimization: bool = True,
    convert_zero_size_to_none: bool = False,
) -> tf.data.Dataset:
  """Makes a TensorFlow dataset.

  We need to explicitly specify up-front the shapes and dtypes of all the
  Tensors that will be drawn from the dataset. We require that the action and
  observation specs are given. The reward and discount specs use reasonable
  defaults if not given. We can also specify a boolean `transition_adder` which
  if true will specify the spec as transitions rather than timesteps (i.e. they
  have a trailing state). Additionally an `extra_spec` parameter can be given
  which specifies "extra data".

  Args:
    client: A TFClient (or list of TFClients) for talking to a replay server.
    environment_spec: The environment's spec.
    batch_size: Optional. If specified the dataset returned will combine
      consecutive elements into batches. This argument is also used to determine
      the cycle_length for `tf.data.Dataset.interleave` -- if unspecified the
      cycle length is set to `tf.data.experimental.AUTOTUNE`.
    prefetch_size: How many batches to prefectch in the pipeline.
    sequence_length: Optional. If specified consecutive elements of each
      interleaved dataset will be combined into sequences.
    extra_spec: Optional. A possibly nested structure of specs for extras. Note
      that whether or not this is present changes the format of the data.
    transition_adder: Optional, defaults to False; whether the adder used with
      this dataset adds transitions.
    table: The name of the table to sample from replay (defaults to
      `adders.DEFAULT_PRIORITY_TABLE`).
    parallel_batch_optimization: Whether to enable the parallel_batch
      optimization. In some cases this optimization may slow down sampling from
      the dataset, in which case turning this to False may speed up performance.
    convert_zero_size_to_none: When True this will convert specs with shapes 0
      to None. This is useful for datasets that contain elements with different
      shapes for example `GraphsTuple` from the graph_net library. For example,
      `specs.Array((0, 5), tf.float32)` will correspond to a examples with shape
      `tf.TensorShape([None, 5])`.

  Returns:
    A tf.data.Dataset that streams data from the replay server.
  """

  assert isinstance(client, reverb.TFClient)

  # Use the environment spec but convert it to a plain tuple.
  adder_spec = tuple(environment_spec)

  # The *transition* adder is special in that it also adds an arrival state.
  if transition_adder:
    adder_spec += (environment_spec.observations,)

  # Any 'extra' data that is passed to the adder is put on the end.
  if extra_spec:
    adder_spec += (extra_spec,)

  # Extract the shapes and dtypes from these specs.
  get_dtype = lambda x: tf.as_dtype(x.dtype)
  get_shape = lambda x: tf.TensorShape(x.shape)
  if sequence_length:
    get_shape = lambda x: tf.TensorShape([sequence_length, *x.shape])

  if convert_zero_size_to_none:
    # TODO(b/143692455): Consider making this default behaviour.
    get_shape = lambda x: tf.TensorShape([s if s else None for s in x.shape])
  shapes = tree.map_structure(get_shape, adder_spec)
  dtypes = tree.map_structure(get_dtype, adder_spec)

  def _make_dataset(unused_idx: tf.Tensor) -> tf.data.Dataset:
    dataset = client.dataset(
        table=table,
        dtypes=dtypes,
        shapes=shapes,
        capacity=2 * batch_size if batch_size else 100,
        sequence_length=sequence_length,
        emit_timesteps=sequence_length is None,
    )
    return dataset

  # Create the dataset.
  dataset = tf.data.Dataset.range(1).repeat()
  dataset = dataset.interleave(
      map_func=_make_dataset,
      cycle_length=batch_size or tf.data.experimental.AUTOTUNE,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # Optimization options.
  options = tf.data.Options()
  options.experimental_deterministic = False
  options.experimental_optimization.parallel_batch = parallel_batch_optimization
  dataset = dataset.with_options(options)

  # Finish the pipeline: batch and prefetch.
  if batch_size:
    dataset = dataset.batch(batch_size, drop_remainder=True)
  if prefetch_size:
    dataset = dataset.prefetch(prefetch_size)

  return dataset


# TODO(b/152732834): remove this and prefer datasets.make_reverb_dataset.
make_dataset = make_reverb_dataset
