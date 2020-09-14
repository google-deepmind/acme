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

# pylint: disable=g-import-not-at-top
try:
  from acme.adders.reverb.deprecated import base as deprecated_base  # pytype: disable=import-error
except ImportError:
  deprecated_base = None
# pylint: enable=g-import-not-at-top


def make_reverb_dataset(
    server_address: str,
    environment_spec: Optional[specs.EnvironmentSpec] = None,
    batch_size: Optional[int] = None,
    prefetch_size: Optional[int] = None,
    sequence_length: Optional[int] = None,
    extra_spec: Optional[types.NestedSpec] = None,
    transition_adder: bool = False,
    table: str = adders.DEFAULT_PRIORITY_TABLE,
    convert_zero_size_to_none: bool = False,
    num_parallel_calls: int = 16,
    using_deprecated_adder: bool = False,
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
    server_address: The Reverb server's address.
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
    convert_zero_size_to_none: When True this will convert specs with shapes 0
      to None. This is useful for datasets that contain elements with different
      shapes for example `GraphsTuple` from the graph_net library. For example,
      `specs.Array((0, 5), tf.float32)` will correspond to a examples with shape
      `tf.TensorShape([None, 5])`.
    num_parallel_calls: Number of parallel threads creating ReplayDatasets to
      interleave.
    using_deprecated_adder: True if the adder used to generate the data is
      from acme/adders/reverb/deprecated.

  Returns:
    A tf.data.Dataset that streams data from the replay server.
  """

  # This is the default that used to be set by reverb.TFClient.dataset().
  max_in_flight_samples_per_worker = 2 * batch_size if batch_size else 100

  def _make_dataset(unused_idx: tf.Tensor) -> tf.data.Dataset:
    if environment_spec is not None:
      shapes, dtypes = _spec_to_shapes_and_dtypes(
          transition_adder,
          environment_spec,
          extra_spec=extra_spec,
          sequence_length=sequence_length,
          convert_zero_size_to_none=convert_zero_size_to_none,
          using_deprecated_adder=using_deprecated_adder)
      dataset = reverb.ReplayDataset(
          server_address=server_address,
          table=table,
          dtypes=dtypes,
          shapes=shapes,
          max_in_flight_samples_per_worker=max_in_flight_samples_per_worker,
          sequence_length=sequence_length,
          emit_timesteps=sequence_length is None)
    else:
      dataset = reverb.ReplayDataset.from_table_signature(
          server_address=server_address,
          table=table,
          max_in_flight_samples_per_worker=max_in_flight_samples_per_worker,
          sequence_length=sequence_length,
          emit_timesteps=sequence_length is None)

    # Finish the pipeline: batch and prefetch.
    if batch_size:
      dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset

  # Create the dataset.
  dataset = tf.data.Dataset.range(num_parallel_calls)
  dataset = dataset.interleave(
      map_func=_make_dataset,
      cycle_length=num_parallel_calls,
      num_parallel_calls=num_parallel_calls,
      deterministic=False)

  if prefetch_size:
    dataset = dataset.prefetch(prefetch_size)

  return dataset


def _spec_to_shapes_and_dtypes(transition_adder: bool,
                               environment_spec: specs.EnvironmentSpec,
                               extra_spec: Optional[types.NestedSpec],
                               sequence_length: Optional[int],
                               convert_zero_size_to_none: bool,
                               using_deprecated_adder: bool):
  """Creates the shapes and dtypes needed to describe the Reverb dataset.

  This takes a `environment_spec`, `extra_spec`, and additional information and
  returns a tuple (shapes, dtypes) that describe the data contained in Reverb.

  Args:
    transition_adder: A boolean, describing if a `TransitionAdder` was used to
      add data.
    environment_spec: A `specs.EnvironmentSpec`, describing the shapes and
      dtypes of the data produced by the environment (and the action).
    extra_spec: A nested structure of objects with a `.shape` and `.dtype`
      property. This describes any additional data the Actor adds into Reverb.
    sequence_length: An optional integer for how long the added sequences are,
      only used with `SequenceAdder`.
    convert_zero_size_to_none: If True, then all shape dimensions that are 0 are
      converted to None. A None dimension is only set at runtime.
    using_deprecated_adder: True if the adder used to generate the data is
      from acme/adders/reverb/deprecated.

  Returns:
    A tuple (dtypes, shapes) that describes the data that has been added into
    Reverb.
  """
  # The *transition* adder is special in that it also adds an arrival state.
  if transition_adder:
    # Use the environment spec but convert it to a plain tuple.
    adder_spec = tuple(environment_spec) + (environment_spec.observations,)
    # Any 'extra' data that is passed to the adder is put on the end.
    if extra_spec:
      adder_spec += (extra_spec,)
  elif using_deprecated_adder and deprecated_base is not None:
    adder_spec = deprecated_base.Step(
        observation=environment_spec.observations,
        action=environment_spec.actions,
        reward=environment_spec.rewards,
        discount=environment_spec.discounts,
        extras=() if not extra_spec else extra_spec)
  else:
    adder_spec = adders.Step(
        observation=environment_spec.observations,
        action=environment_spec.actions,
        reward=environment_spec.rewards,
        discount=environment_spec.discounts,
        start_of_episode=specs.Array(shape=(), dtype=bool),
        extras=() if not extra_spec else extra_spec)

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
  return shapes, dtypes


# TODO(b/152732834): remove this and prefer datasets.make_reverb_dataset.
make_dataset = make_reverb_dataset
