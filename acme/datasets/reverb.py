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

import os
from typing import Callable, Optional

from acme import specs
from acme import types
from acme.adders import reverb as adders
import reverb
import tensorflow as tf

Transform = Callable[[reverb.ReplaySample], reverb.ReplaySample]


def make_reverb_dataset(
    server_address: str,
    batch_size: Optional[int] = None,
    prefetch_size: Optional[int] = None,
    table: str = adders.DEFAULT_PRIORITY_TABLE,
    num_parallel_calls: Optional[int] = 12,
    max_in_flight_samples_per_worker: Optional[int] = None,
    postprocess: Optional[Transform] = None,
    # Deprecated kwargs.
    environment_spec: Optional[specs.EnvironmentSpec] = None,
    extra_spec: Optional[types.NestedSpec] = None,
    transition_adder: bool = False,
    convert_zero_size_to_none: bool = False,
    using_deprecated_adder: bool = False,
    sequence_length: Optional[int] = None,
) -> tf.data.Dataset:
  """Make a TensorFlow dataset backed by a Reverb trajectory replay service.

  Arguments:
    server_address: Address of the Reverb server.
    batch_size: Batch size of the returned dataset.
    prefetch_size: The number of elements to prefetch from the original dataset.
      Note that Reverb may do some internal prefetching in addition to this.
    table: The name of the Reverb table to use.
    num_parallel_calls: The parralelism to use. Setting it to `tf.data.AUTOTUNE`
      will allow `tf.data` to automatically find a reasonable value.
    max_in_flight_samples_per_worker: see reverb.TrajectoryDataset for details.
    postprocess: User-specified transformation to be applied to the dataset (as
      `ds.map(postprocess)`).
    environment_spec: DEPRECATED! Do not use.
    extra_spec: DEPRECATED! Do not use.
    transition_adder: DEPRECATED! Do not use.
    convert_zero_size_to_none: DEPRECATED! Do not use.
    using_deprecated_adder: DEPRECATED! Do not use.
    sequence_length: DEPRECATED! Do not use.

  Returns:
    A `tf.data.Dataset` iterating over the contents of the Reverb table.

  Raises:
    ValueError if `environment_spec` or `extra_spec` are set.
  """

  if environment_spec or extra_spec:
    raise ValueError(
        'The make_reverb_dataset factory function no longer requires specs as'
        ' as they should be passed as a signature to the reverb.Table when it'
        ' is created. Consider either updating your code or falling back to the'
        ' deprecated dataset factory in acme/datasets/deprecated.')

  # These are no longer used and are only kept in the call signature for
  # backward compatibility.
  del environment_spec
  del extra_spec
  del transition_adder
  del convert_zero_size_to_none
  del using_deprecated_adder
  del sequence_length

  # This is the default that used to be set by reverb.TFClient.dataset().
  if max_in_flight_samples_per_worker is None and batch_size is None:
    max_in_flight_samples_per_worker = 100
  elif max_in_flight_samples_per_worker is None:
    max_in_flight_samples_per_worker = 2 * batch_size

  def _make_dataset(unused_idx: tf.Tensor) -> tf.data.Dataset:
    dataset = reverb.TrajectoryDataset.from_table_signature(
        server_address=server_address,
        table=table,
        max_in_flight_samples_per_worker=max_in_flight_samples_per_worker)

    # Post-process each element if a post-processing function is passed, e.g.
    # observation-stacking or data augmenting transformations.
    if postprocess:
      dataset = dataset.map(postprocess)

    if batch_size:
      dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset

  if num_parallel_calls is not None:
    # Create a datasets and interleaves it to create `num_parallel_calls`
    # `TrajectoryDataset`s.
    num_datasets_to_interleave = (
        os.cpu_count()
        if num_parallel_calls == tf.data.AUTOTUNE else num_parallel_calls)
    dataset = tf.data.Dataset.range(num_datasets_to_interleave).interleave(
        map_func=_make_dataset,
        cycle_length=num_parallel_calls,
        num_parallel_calls=num_parallel_calls,
        deterministic=False)
  else:
    dataset = _make_dataset(tf.constant(0))

  if prefetch_size:
    dataset = dataset.prefetch(prefetch_size)

  return dataset
