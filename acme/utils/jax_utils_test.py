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

"""Tests for jax_utils."""

from absl.testing import absltest
from acme.utils import jax_utils

import jax.numpy as jnp


class JaxUtilsTest(absltest.TestCase):

  def test_batch_concat(self):
    batch_size = 32
    inputs = [
        jnp.zeros(shape=(batch_size, 2)),
        {
            'foo': jnp.zeros(shape=(batch_size, 5, 3))
        },
        [jnp.zeros(shape=(batch_size, 1))],
    ]

    output_shape = jax_utils.batch_concat(inputs).shape
    expected_shape = [batch_size, 2 + 5 * 3 + 1]
    self.assertSequenceEqual(output_shape, expected_shape)


if __name__ == '__main__':
  absltest.main()
