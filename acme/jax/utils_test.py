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

"""Tests for utils."""

from absl.testing import absltest
from acme.jax import utils

import chex
import jax
import jax.numpy as jnp
import numpy as np

chex.set_n_cpu_devices(4)


class JaxUtilsTest(absltest.TestCase):

  def test_batch_concat(self):
    batch_size = 32
    inputs = [
        jnp.zeros(shape=(batch_size, 2)),
        {
            'foo': jnp.zeros(shape=(batch_size, 5, 3))
        },
        [jnp.zeros(shape=(batch_size, 1))],
        jnp.zeros(shape=(batch_size,)),
    ]

    output_shape = utils.batch_concat(inputs).shape
    expected_shape = [batch_size, 2 + 5 * 3 + 1 + 1]
    self.assertSequenceEqual(output_shape, expected_shape)

  def test_mapreduce(self):

    @utils.mapreduce
    def f(y, x):
      return jnp.square(x + y)

    z = f(jnp.ones(shape=(32,)), jnp.ones(shape=(32,)))
    z = jax.device_get(z)
    self.assertEqual(z, 4)

  def test_get_from_first_device(self):
    sharded = {
        'a':
            jax.device_put_sharded(
                list(jnp.arange(16).reshape([jax.local_device_count(), 4])),
                jax.local_devices()),
        'b':
            jax.device_put_sharded(
                list(jnp.arange(8).reshape([jax.local_device_count(), 2])),
                jax.local_devices(),
            ),
    }

    want = {
        'a': jnp.arange(4),
        'b': jnp.arange(2),
    }

    # Get zeroth device content as DeviceArray.
    device_arrays = utils.get_from_first_device(sharded, as_numpy=False)
    jax.tree_map(
        lambda x: self.assertIsInstance(x, jax.xla.DeviceArray),
        device_arrays)
    jax.tree_map(np.testing.assert_array_equal, want, device_arrays)

    # Get the zeroth device content as numpy arrays.
    numpy_arrays = utils.get_from_first_device(sharded, as_numpy=True)
    jax.tree_map(lambda x: self.assertIsInstance(x, np.ndarray), numpy_arrays)
    jax.tree_map(np.testing.assert_array_equal, want, numpy_arrays)

  def test_get_from_first_device_fails_if_sda_not_provided(self):
    with self.assertRaises(ValueError):
      utils.get_from_first_device({'a': np.arange(jax.local_device_count())})


if __name__ == '__main__':
  absltest.main()
