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

"""Tests for running statistics utilities."""

import functools
import math

from absl.testing import absltest
from acme import specs
from acme.jax import running_statistics
from acme.jax import utils
import jax
from jax import test_util as jtu
from jax.config import config as jax_config
import jax.numpy as jnp
import tree

update_and_validate = functools.partial(
    running_statistics.update, validate_shapes=True)


class RunningStatisticsTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    jax_config.update('jax_enable_x64', False)

  def test_normalize(self):
    state = running_statistics.init_state(specs.Array((5,), jnp.float32))

    x = jnp.arange(200, dtype=jnp.float32).reshape(20, 2, 5)
    x1, x2, x3, x4 = jnp.split(x, 4, axis=0)

    state = update_and_validate(state, x1, axis=(0, 1))
    state = update_and_validate(state, x2, axis=(0, 1))
    state = update_and_validate(state, x3, axis=(0, 1))
    state = update_and_validate(state, x4, axis=(0, 1))
    normalized = running_statistics.normalize(x, state)

    mean = jnp.mean(normalized, axis=(0, 1))
    std = jnp.std(normalized, axis=(0, 1))
    self.assertAllClose(mean, jnp.zeros_like(mean))
    self.assertAllClose(std, jnp.ones_like(std))

  def test_init_normalize(self):
    state = running_statistics.init_state(specs.Array((5,), jnp.float32))

    x = jnp.arange(200, dtype=jnp.float32).reshape(20, 2, 5)
    normalized = running_statistics.normalize(x, state)

    self.assertAllClose(normalized, x)

  def test_axis_none(self):
    state = running_statistics.init_state(specs.Array((), jnp.float32))

    x = jnp.arange(5, dtype=jnp.float32)

    state = update_and_validate(state, x)
    normalized = running_statistics.normalize(x, state)

    mean = jnp.mean(normalized)
    std = jnp.std(normalized)
    self.assertAllClose(mean, jnp.zeros_like(mean))
    self.assertAllClose(std, jnp.ones_like(std))

  def test_axis_empty(self):
    state = running_statistics.init_state(specs.Array((5,), jnp.float32))

    x = jnp.arange(5, dtype=jnp.float32)

    state = update_and_validate(state, x, axis=())
    normalized = running_statistics.normalize(x, state)

    mean = jnp.mean(normalized, axis=())
    std = jnp.std(normalized, axis=())
    self.assertAllClose(mean, jnp.zeros_like(mean))
    self.assertAllClose(std, jnp.zeros_like(std))

  def test_one_batch_dim(self):
    state = running_statistics.init_state(specs.Array((5,), jnp.float32))

    x = jnp.arange(10, dtype=jnp.float32).reshape(2, 5)

    state = update_and_validate(state, x, axis=0)
    normalized = running_statistics.normalize(x, state)

    mean = jnp.mean(normalized, axis=0)
    std = jnp.std(normalized, axis=0)
    self.assertAllClose(mean, jnp.zeros_like(mean))
    self.assertAllClose(std, jnp.ones_like(std))

  def test_clip(self):
    state = running_statistics.init_state(specs.Array((), jnp.float32))

    x = jnp.arange(5, dtype=jnp.float32)

    state = update_and_validate(state, x)
    normalized = running_statistics.normalize(x, state, max_abs_value=1.0)

    mean = jnp.mean(normalized)
    std = jnp.std(normalized)
    self.assertAllClose(mean, jnp.zeros_like(mean))
    self.assertAllClose(std, jnp.ones_like(std) * math.sqrt(0.6))

  def test_int32_overflow(self):
    state = running_statistics.init_state(specs.Array((2,), jnp.float32))

    # Batch size is 2 * 65536.
    x = jnp.arange(4, dtype=jnp.float32).reshape(2, 2)
    batch = utils.tile_array(x, 65536)

    update = jax.jit(
        functools.partial(update_and_validate, axis=(0, 1)), backend='cpu')

    for _ in range(16383):
      state = update(state, batch)

    normalized = running_statistics.normalize(x, state)

    # We added 16383 batches, so the count is 2*65536*16383 = 2147352576.
    # This fits in int32s.
    mean = jnp.mean(normalized, axis=0)
    std = jnp.std(normalized, axis=0)
    self.assertAllClose(mean, jnp.zeros_like(mean))
    self.assertAllClose(std, jnp.ones_like(std))

    overflow_state = update(state, batch)
    # Added one more batch, now the count value is 2*65536*16384 = 2147483648,
    # which is max_int32s+1. Expecting overflow.
    self.assertLess(overflow_state.count, 0)

    # it should work correctly if recompiled in 64-bit mode.
    jax_config.update('jax_enable_x64', True)
    update = jax.jit(
        functools.partial(update_and_validate, axis=(0, 1)), backend='cpu')
    state = update(state, batch)
    # With jax_enable_x64, the the count is promoted to int64s, so we can safely
    # add one more batch without overflow.
    self.assertEqual(state.count, 2 * 65536 * 16384)
    normalized = running_statistics.normalize(x, state)

    mean = jnp.mean(normalized, axis=0)
    std = jnp.std(normalized, axis=0)
    self.assertAllClose(mean, jnp.zeros_like(mean))
    self.assertAllClose(std, jnp.ones_like(std))
    jax_config.update('jax_enable_x64', False)

  def test_nested_normalize(self):
    state = running_statistics.init_state({
        'a': specs.Array((5,), jnp.float32),
        'b': specs.Array((2,), jnp.float32)
    })

    x1 = {
        'a': jnp.arange(20, dtype=jnp.float32).reshape(2, 2, 5),
        'b': jnp.arange(8, dtype=jnp.float32).reshape(2, 2, 2)
    }
    x2 = {
        'a': jnp.arange(20, dtype=jnp.float32).reshape(2, 2, 5) + 20,
        'b': jnp.arange(8, dtype=jnp.float32).reshape(2, 2, 2) + 8
    }
    x3 = {
        'a': jnp.arange(40, dtype=jnp.float32).reshape(4, 2, 5),
        'b': jnp.arange(16, dtype=jnp.float32).reshape(4, 2, 2)
    }

    state = update_and_validate(state, x1, axis=(0, 1))
    state = update_and_validate(state, x2, axis=(0, 1))
    state = update_and_validate(state, x3, axis=(0, 1))
    normalized = running_statistics.normalize(x3, state)

    mean = tree.map_structure(lambda x: jnp.mean(x, axis=(0, 1)), normalized)
    std = tree.map_structure(lambda x: jnp.std(x, axis=(0, 1)), normalized)
    tree.map_structure(lambda x: self.assertAllClose(x, jnp.zeros_like(x)),
                       mean)
    tree.map_structure(lambda x: self.assertAllClose(x, jnp.ones_like(x)), std)

  def test_validation(self):
    state = running_statistics.init_state(specs.Array((1, 2, 3), jnp.float32))

    x = jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3)
    with self.assertRaises(AssertionError):
      update_and_validate(state, x, axis=())

    x = jnp.arange(3, dtype=jnp.float32).reshape(1, 1, 3)
    with self.assertRaises(AssertionError):
      update_and_validate(state, x, axis=())

  def test_int_not_normalized(self):
    state = running_statistics.init_state(specs.Array((), jnp.int32))

    x = jnp.arange(5, dtype=jnp.int32)

    state = update_and_validate(state, x)
    normalized = running_statistics.normalize(x, state)

    self.assertArraysEqual(normalized, x)

if __name__ == '__main__':
  absltest.main()
