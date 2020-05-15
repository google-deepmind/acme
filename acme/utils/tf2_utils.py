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

"""Utilities for nested data structures involving NumPy and TensorFlow 2.x."""

import functools
from typing import List, Optional, Sequence, TypeVar

from acme import types

import numpy as np
import sonnet as snt
import tensorflow as tf
import tree

SequenceType = TypeVar('SequenceType')


def add_batch_dim(nest: types.NestedTensor) -> types.NestedTensor:
  """Adds a batch dimension to each leaf of a nested structure of Tensors."""
  return tree.map_structure(lambda x: tf.expand_dims(x, axis=0), nest)


def squeeze_batch_dim(nest: types.NestedTensor) -> types.NestedTensor:
  """Squeezes out a batch dimension from each leaf of a nested structure."""
  return tree.map_structure(lambda x: tf.squeeze(x, axis=0), nest)


def batch_concat(inputs: types.NestedTensor) -> tf.Tensor:
  """Concatenate a collection of Tensors while preserving the batch dimension.

  This takes a potentially nested collection of tensors, flattens everything
  but the batch (first) dimension, and concatenates along the resulting data
  (second) dimension.

  Args:
    inputs: a tensor or nested collection of tensors.

  Returns:
    A concatenated tensor which maintains the batch dimension but concatenates
    all other data along the flattened second dimension.
  """
  flat_leaves = tree.map_structure(snt.Flatten(), inputs)
  return tf.concat(tree.flatten(flat_leaves), axis=-1)


def batch_to_sequence(data: types.NestedTensor) -> types.NestedTensor:
  """Converts data between sequence-major and batch-major format."""
  return tree.map_structure(
      lambda t: tf.transpose(t, [1, 0] + list(range(2, t.shape.rank))), data)


def tile_tensor(tensor: tf.Tensor, multiple: int) -> tf.Tensor:
  """Tiles `multiple` copies of `tensor` along a new leading axis."""
  rank = len(tensor.shape)
  multiples = tf.constant([multiple] + [1] * rank, dtype=tf.int32)
  expanded_tensor = tf.expand_dims(tensor, axis=0)
  return tf.tile(expanded_tensor, multiples)


def tile_nested(inputs: types.NestedTensor,
                multiple: int) -> types.NestedTensor:
  """Tiles tensors in a nested structure along a new leading axis."""
  tile = functools.partial(tile_tensor, multiple=multiple)
  return tree.map_structure(tile, inputs)


def create_variables(
    network: snt.Module,
    input_spec: List[types.NestedSpec],
) -> Optional[tf.TensorSpec]:
  """Builds the network with dummy inputs to create the necessary variables.

  Args:
    network: Sonnet Module whose variables are to be created.
    input_spec: list of input specs to the network. The length of this list
      should match the number of arguments expected by `network`.

  Returns:
    output_spec: only returns an output spec if the output is a tf.Tensor, else
        it doesn't return anything (None); e.g. if the output is a
        tfp.distributions.Distribution.
  """
  # Create a dummy observation with no batch dimension.
  dummy_input = zeros_like(input_spec)

  # If we have an RNNCore the hidden state will be an additional input.
  if isinstance(network, snt.RNNCore):
    initial_state = squeeze_batch_dim(network.initial_state(1))
    dummy_input += [initial_state]

  # Forward pass of the network which will create variables as a side effect.
  dummy_output = network(*add_batch_dim(dummy_input))

  # Evaluate the input signature by converting the dummy input into a
  # TensorSpec. We then save the signature as a property of the network. This is
  # done so that we can later use it when creating snapshots. We do this here
  # because the snapshot code may not have access to the precise form of the
  # inputs.
  input_signature = tree.map_structure(
      lambda t: tf.TensorSpec((None,) + t.shape, t.dtype), dummy_input)
  network._input_signature = input_signature  # pylint: disable=protected-access

  def spec(output):
    # If the output is not a Tensor, return None as spec is ill-defined.
    if not isinstance(output, tf.Tensor):
      return None
    # If this is not a scalar Tensor, make sure to squeeze out the batch dim.
    if tf.rank(output) > 0:
      output = squeeze_batch_dim(output)
    return tf.TensorSpec(output.shape, output.dtype)

  return tree.map_structure(spec, dummy_output)


def to_sonnet_module(transformation: types.TensorTransformation) -> snt.Module:
  """Convert a tensor transformation to a Sonnet Module."""

  if isinstance(transformation, snt.Module):
    return transformation

  # Use snt.Sequential to convert any tensor transformation to a snt.Module.
  module = snt.Sequential([transformation])

  # Wrap the module to allow it to return an empty variable tuple.
  return snt.allow_empty_variables(module)


def to_numpy(nest: types.NestedTensor) -> types.NestedArray:
  """Converts a nest of Tensors to a nest of numpy arrays."""
  return tree.map_structure(lambda x: x.numpy(), nest)


def to_numpy_squeeze(nest: types.NestedTensor, axis=0) -> types.NestedArray:
  """Converts a nest of Tensors to a nest of numpy arrays and squeeze axis."""
  return tree.map_structure(lambda x: x.numpy().squeeze(axis=axis), nest)


def zeros_like(nest: types.Nest) -> types.NestedTensor:
  """Given a nest of array-like objects, returns similarly nested tf.zeros."""
  return tree.map_structure(lambda x: tf.zeros(x.shape, x.dtype), nest)


def fast_map_structure(func, *structure):
  """Faster map_structure implementation which skips some error checking."""
  flat_structure = (tree.flatten(s) for s in structure)
  entries = zip(*flat_structure)
  # Arbitrarily choose one of the structures of the original sequence (the last)
  # to match the structure for the flattened sequence.
  return tree.unflatten_as(structure[-1], [func(*x) for x in entries])


def stack_sequence_fields(sequence: Sequence[SequenceType]) -> SequenceType:
  """Stacks a list of identically nested objects.

  This takes a sequence of identically nested objects and returns a single
  nested object whose ith leaf is a stacked numpy array of the corresponding
  ith leaf from each element of the sequence.

  For example, if `sequence` is:

  ```python
  [{
        'action': np.array([1.0]),
        'observation': (np.array([0.0, 1.0, 2.0]),),
        'reward': 1.0
   }, {
        'action': np.array([0.5]),
        'observation': (np.array([1.0, 2.0, 3.0]),),
        'reward': 0.0
   }, {
        'action': np.array([0.3]),
        'observation': (np.array([2.0, 3.0, 4.0]),),
        'reward': 0.5
   }]
  ```

  Then this function will return:

  ```python
  {
      'action': np.array([....])         # array shape = [3 x 1]
      'observation': (np.array([...]),)  # array shape = [3 x 3]
      'reward': np.array([...])          # array shape = [3]
  }
  ```

  Note that the 'observation' entry in the above example has two levels of
  nesting, i.e it is a tuple of arrays.

  Args:
    sequence: a list of identically nested objects.

  Returns:
    A nested object with numpy.

  Raises:
    ValueError: If `sequence` is an empty sequence.
  """
  # Handle empty input sequences.
  if not sequence:
    raise ValueError('Input sequence must not be empty')

  return fast_map_structure(lambda *values: np.asarray(values), *sequence)
