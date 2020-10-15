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

"""Tests for acme.utils.loggers.base."""

from absl.testing import absltest
from acme.utils.loggers import base
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


class BaseTest(absltest.TestCase):

  def test_tensor_serialisation(self):
    data = {'x': tf.zeros(shape=(32,))}
    output = base.to_numpy(data)
    expected = {'x': np.zeros(shape=(32,))}
    np.testing.assert_array_equal(output['x'], expected['x'])

  def test_device_array_serialisation(self):
    data = {'x': jnp.zeros(shape=(32,))}
    output = base.to_numpy(data)
    expected = {'x': np.zeros(shape=(32,))}
    np.testing.assert_array_equal(output['x'], expected['x'])


if __name__ == '__main__':
  absltest.main()
