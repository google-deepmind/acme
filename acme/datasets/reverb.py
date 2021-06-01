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


def make_reverb_dataset(
    server_address: str,
    batch_size: Optional[int] = None,
    prefetch_size: Optional[int] = None,
    table: str = adders.DEFAULT_PRIORITY_TABLE,
    num_parallel_calls: int = 12,
    max_in_flight_samples_per_worker: Optional[int] = None,
    # Deprecated kwargs.
    environment_spec: Optional[specs.EnvironmentSpec] = None,
    extra_spec: Optional[types.NestedSpec] = None,
    transition_adder: bool = False,
    convert_zero_size_to_none: bool = False,
    using_deprecated_adder: bool = False,
    sequence_length: Optional[int] = None,
) -> tf.data.Dataset:
  """Make a TensorFlow dataset backed by a Reverb trajectory replay service."""
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


# TODO(b/152732834): remove this and prefer datasets.make_reverb_dataset.
make_dataset = make_reverb_dataset
