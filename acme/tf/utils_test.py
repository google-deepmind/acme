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

"""Tests for acme.tf.utils."""

from typing import Sequence, Tuple

from absl.testing import absltest
from absl.testing import parameterized

from acme import specs
from acme.tf import utils as tf2_utils

import numpy as np
import sonnet as snt
import tensorflow as tf


class PolicyValueHead(snt.Module):
  """A network with two linear layers, for policy and value respectively."""

  def __init__(self, num_actions: int):
    super().__init__(name='policy_value_network')
    self._policy_layer = snt.Linear(num_actions)
    self._value_layer = snt.Linear(1)

  def __call__(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Returns a (Logits, Value) tuple."""
    logits = self._policy_layer(inputs)  # [B, A]
    value = tf.squeeze(self._value_layer(inputs), axis=-1)  # [B]

    return logits, value


class CreateVariableTest(parameterized.TestCase):
  """Tests for tf2_utils.create_variables method."""

  @parameterized.parameters([True, False])
  def test_feedforward(self, recurrent: bool):
    model = snt.Linear(42)
    if recurrent:
      model = snt.DeepRNN([model])
    input_spec = specs.Array(shape=(10,), dtype=np.float32)
    tf2_utils.create_variables(model, [input_spec])
    variables: Sequence[tf.Variable] = model.variables
    shapes = [v.shape.as_list() for v in variables]
    self.assertSequenceEqual(shapes, [[42], [10, 42]])

  @parameterized.parameters([True, False])
  def test_output_spec_feedforward(self, recurrent: bool):
    input_spec = specs.Array(shape=(10,), dtype=np.float32)
    model = snt.Linear(42)
    expected_spec = tf.TensorSpec(shape=(42,), dtype=tf.float32)
    if recurrent:
      model = snt.DeepRNN([model])
      expected_spec = (expected_spec, ())

    output_spec = tf2_utils.create_variables(model, [input_spec])
    self.assertEqual(output_spec, expected_spec)

  def test_multiple_ouputs(self):
    model = PolicyValueHead(42)
    input_spec = specs.Array(shape=(10,), dtype=np.float32)
    expected_spec = (tf.TensorSpec(shape=(42,), dtype=tf.float32),
                     tf.TensorSpec(shape=(), dtype=tf.float32))
    output_spec = tf2_utils.create_variables(model, [input_spec])
    variables: Sequence[tf.Variable] = model.variables
    shapes = [v.shape.as_list() for v in variables]
    self.assertSequenceEqual(shapes, [[42], [10, 42], [1], [10, 1]])
    self.assertSequenceEqual(output_spec, expected_spec)

  def test_scalar_output(self):
    model = tf2_utils.to_sonnet_module(tf.reduce_sum)
    input_spec = specs.Array(shape=(10,), dtype=np.float32)
    expected_spec = tf.TensorSpec(shape=(), dtype=tf.float32)
    output_spec = tf2_utils.create_variables(model, [input_spec])
    self.assertEqual(model.variables, ())
    self.assertEqual(output_spec, expected_spec)

  def test_none_output(self):
    model = tf2_utils.to_sonnet_module(lambda x: None)
    input_spec = specs.Array(shape=(10,), dtype=np.float32)
    expected_spec = None
    output_spec = tf2_utils.create_variables(model, [input_spec])
    self.assertEqual(model.variables, ())
    self.assertEqual(output_spec, expected_spec)


class Tf2UtilsTest(parameterized.TestCase):
  """Tests for tf2_utils methods."""

  def test_batch_concat(self):
    batch_size = 32
    inputs = [
        tf.zeros(shape=(batch_size, 2)),
        {
            'foo': tf.zeros(shape=(batch_size, 5, 3))
        },
        [tf.zeros(shape=(batch_size, 1))],
    ]

    output_shape = tf2_utils.batch_concat(inputs).shape.as_list()
    expected_shape = [batch_size, 2 + 5 * 3 + 1]
    self.assertSequenceEqual(output_shape, expected_shape)

  def test_stack_sequence_fields(self):
    sequence = [{
        'action': np.array([1.0]),
        'observation': (np.array([0.0, 1.0, 2.0]),),
        'reward': np.array(1.0),
    }, {
        'action': np.array([0.5]),
        'observation': (np.array([1.0, 2.0, 3.0]),),
        'reward': np.array(0.0),
    }, {
        'action': np.array([0.3]),
        'observation': (np.array([2.0, 3.0, 4.0]),),
        'reward': np.array(0.5),
    }]

    stacked = tf2_utils.stack_sequence_fields(sequence)

    self.assertIsInstance(stacked, dict)
    self.assertLen(stacked.keys(), 3)
    self.assertLen(stacked['observation'], 1)

    self.assertEqual(stacked['action'].shape, (3, 1))
    self.assertEqual(stacked['observation'][0].shape, (3, 3))
    self.assertEqual(stacked['reward'].shape, (3,))
    self.assertEqual(stacked['observation'][0].tolist(),
                     [[0., 1., 2.], [1., 2., 3.], [2., 3., 4.]])


if __name__ == '__main__':
  absltest.main()
