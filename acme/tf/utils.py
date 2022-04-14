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
from typing import List, Optional, Union

from acme import types
from acme.utils import tree_utils

import sonnet as snt
import tensorflow as tf
import tree


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
    input_spec: List[Union[types.NestedSpec, tf.TensorSpec]],
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


class TransformationWrapper(snt.Module):
  """Helper class for to_sonnet_module.

  This wraps arbitrary Tensor-valued callables as a Sonnet module.
  A use case for this is in agent code that could take either a trainable
  sonnet module or a hard-coded function as its policy. By wrapping a hard-coded
  policy with this class, the agent can then treat it as if it were a Sonnet
  module. This removes the need for "if is_hard_coded:..." branches, which you'd
  otherwise need if e.g. calling get_variables() on the policy.
  """

  def __init__(self,
               transformation: types.TensorValuedCallable,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._transformation = transformation

  def __call__(self, *args, **kwargs):
    return self._transformation(*args, **kwargs)


def to_sonnet_module(
    transformation: types.TensorValuedCallable
    ) -> snt.Module:
  """Convert a tensor transformation to a Sonnet Module.

  Args:
    transformation: A Callable that takes one or more (nested) Tensors, and
      returns one or more (nested) Tensors.

  Returns:
    A Sonnet Module that wraps the transformation.
  """

  if isinstance(transformation, snt.Module):
    return transformation

  module = TransformationWrapper(transformation)

  # Wrap the module to allow it to return an empty variable tuple.
  return snt.allow_empty_variables(module)


def to_numpy(nest: types.NestedTensor) -> types.NestedArray:
  """Converts a nest of Tensors to a nest of numpy arrays."""
  return tree.map_structure(lambda x: x.numpy(), nest)


def to_numpy_squeeze(nest: types.NestedTensor, axis=0) -> types.NestedArray:
  """Converts a nest of Tensors to a nest of numpy arrays and squeeze axis."""
  return tree.map_structure(lambda x: tf.squeeze(x, axis=axis).numpy(), nest)


def zeros_like(nest: types.Nest) -> types.NestedTensor:
  """Given a nest of array-like objects, returns similarly nested tf.zeros."""
  return tree.map_structure(lambda x: tf.zeros(x.shape, x.dtype), nest)


# TODO(b/160311329): Migrate call-sites and remove.
stack_sequence_fields = tree_utils.stack_sequence_fields
