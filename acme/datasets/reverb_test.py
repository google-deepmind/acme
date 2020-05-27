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

"""Tests for reverb datasets."""

from absl.testing import absltest

from acme import specs
from acme.datasets import reverb as reverb_dataset
from acme.testing import fakes

import numpy as np
import reverb
import tensorflow as tf
import tree


def _reverb_server():
  return reverb.Server(
      tables=[
          reverb.Table(
              'test_table',
              reverb.selectors.Uniform(),
              reverb.selectors.Fifo(),
              max_size=100,
              rate_limiter=reverb.rate_limiters.MinSize(95))
      ],
      port=None)


def _reverb_client(port):
  return reverb.Client('localhost:{}'.format(port))


def _reverb_tf_client(port):
  return reverb.TFClient('localhost:{}'.format(port))


def _check_specs(array_specs, tensor_specs):

  def check_spec(array_spec: specs.Array, tensor_spec: tf.TensorSpec):
    return (np.array_equal(array_spec.shape, tensor_spec.shape) and
            tf.dtypes.as_dtype(array_spec.dtype) == tensor_spec.dtype)

  return tree.map_structure(check_spec, array_specs, tensor_specs)


class DatasetsTest(absltest.TestCase):
  _client: reverb.Client
  _server: reverb.Server

  @classmethod
  def setUpClass(cls):
    super(DatasetsTest, cls).setUpClass()
    cls._server = _reverb_server()
    cls._client = _reverb_client(cls._server.port)

  def setUp(self):
    super(DatasetsTest, self).setUp()
    # We need to create a new reverb.TFClient because the tf.Graph has
    # been reset
    self._tf_client = _reverb_tf_client(self._server.port)

  @property
  def tf_client(self):
    # Use first client for single client tests.
    return self._tf_client

  def tearDown(self):
    super(DatasetsTest, self).tearDown()
    self._client.reset('test_table')

  @classmethod
  def tearDownClass(cls):
    super(DatasetsTest, cls).tearDownClass()
    cls._server.stop()

  def test_make_dataset_simple(self):
    environment = fakes.ContinuousEnvironment()
    environment_spec = specs.make_environment_spec(environment)
    dataset = reverb_dataset.make_dataset(
        client=self.tf_client, environment_spec=environment_spec)

    self.assertTrue(
        _check_specs(tuple(environment_spec), dataset.element_spec.data))

  def test_make_dataset_nested_specs(self):
    environment_spec = specs.EnvironmentSpec(
        observations={
            'obs_1': specs.Array((3, 64, 64), 'uint8'),
            'obs_2': specs.Array((10,), 'int32')
        },
        actions=specs.BoundedArray((), 'float32', minimum=-1., maximum=1.),
        rewards=specs.Array((), 'float32'),
        discounts=specs.BoundedArray((), 'float32', minimum=0., maximum=1.))

    dataset = reverb_dataset.make_dataset(
        client=self.tf_client, environment_spec=environment_spec)

    self.assertTrue(
        _check_specs(tuple(environment_spec), dataset.element_spec.data))

  def test_make_dataset_transition_adder(self):
    environment = fakes.ContinuousEnvironment()
    environment_spec = specs.make_environment_spec(environment)
    dataset = reverb_dataset.make_dataset(
        client=self.tf_client,
        environment_spec=environment_spec,
        transition_adder=True)

    environment_spec = tuple(environment_spec) + (
        environment_spec.observations,)

    self.assertTrue(
        _check_specs(tuple(environment_spec), dataset.element_spec.data))

  def test_make_dataset_with_batch_size(self):
    batch_size = 4
    environment = fakes.ContinuousEnvironment()
    environment_spec = specs.make_environment_spec(environment)
    dataset = reverb_dataset.make_dataset(
        client=self.tf_client,
        environment_spec=environment_spec,
        batch_size=batch_size)

    def make_tensor_spec(spec):
      return tf.TensorSpec(shape=(None,) + spec.shape, dtype=spec.dtype)

    expected_spec = tree.map_structure(make_tensor_spec, environment_spec)

    self.assertTrue(
        _check_specs(tuple(expected_spec), dataset.element_spec.data))

  def test_make_dataset_with_sequence_length_size(self):
    sequence_length = 6
    environment = fakes.ContinuousEnvironment()
    environment_spec = specs.make_environment_spec(environment)
    dataset = reverb_dataset.make_dataset(
        client=self.tf_client,
        environment_spec=environment_spec,
        sequence_length=sequence_length)

    def make_tensor_spec(spec):
      return tf.TensorSpec(
          shape=(sequence_length,) + spec.shape, dtype=spec.dtype)

    expected_spec = tree.map_structure(make_tensor_spec, environment_spec)

    self.assertTrue(
        _check_specs(tuple(expected_spec), dataset.element_spec.data))

  def test_make_dataset_with_sequence_length_and_batch_size(self):
    sequence_length = 6
    batch_size = 4
    environment = fakes.ContinuousEnvironment()
    environment_spec = specs.make_environment_spec(environment)
    dataset = reverb_dataset.make_dataset(
        client=self.tf_client,
        environment_spec=environment_spec,
        batch_size=batch_size,
        sequence_length=sequence_length)

    def make_tensor_spec(spec):
      return tf.TensorSpec(
          shape=(
              batch_size,
              sequence_length,
          ) + spec.shape, dtype=spec.dtype)

    expected_spec = tree.map_structure(make_tensor_spec, environment_spec)

    self.assertTrue(
        _check_specs(tuple(expected_spec), dataset.element_spec.data))

  def test_make_dataset_with_variable_length_instances(self):
    """Dataset with variable length instances should have shapes with None."""
    environment_spec = specs.EnvironmentSpec(
        observations=specs.Array((0, 64, 64), 'uint8'),
        actions=specs.BoundedArray((), 'float32', minimum=-1., maximum=1.),
        rewards=specs.Array((), 'float32'),
        discounts=specs.BoundedArray((), 'float32', minimum=0., maximum=1.))

    dataset = reverb_dataset.make_dataset(
        client=self.tf_client,
        environment_spec=environment_spec,
        convert_zero_size_to_none=True)

    self.assertSequenceEqual(dataset.element_spec.data[0].shape.as_list(),
                             [None, 64, 64])


if __name__ == '__main__':
  absltest.main()
