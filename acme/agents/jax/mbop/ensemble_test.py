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

"""Tests for ensemble."""

import functools
from typing import Any

from acme.agents.jax.mbop import ensemble
from acme.jax import networks
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from absl.testing import absltest


class RandomFFN(nn.Module):

  @nn.compact
  def __call__(self, x):
    return nn.Dense(15)(x)


def params_adding_ffn(x: jnp.ndarray) -> networks.FeedForwardNetwork:
  """Apply adds the parameters to the inputs."""
  return networks.FeedForwardNetwork(
      init=lambda key, x=x: jax.random.uniform(key, x.shape),
      apply=lambda params, x: params + x)


def funny_args_ffn(x: jnp.ndarray) -> networks.FeedForwardNetwork:
  """Apply takes additional parameters, returns `params + x + foo - bar`."""
  return networks.FeedForwardNetwork(
      init=lambda key, x=x: jax.random.uniform(key, x.shape),
      apply=lambda params, x, foo, bar: params + x + foo - bar)


def struct_params_adding_ffn(sx: Any) -> networks.FeedForwardNetwork:
  """Like params_adding_ffn, but with pytree inputs, preserves structure."""

  def init_fn(key, sx=sx):
    return jax.tree_map(lambda x: jax.random.uniform(key, x.shape), sx)

  def apply_fn(params, x):
    return jax.tree_map(lambda p, v: p + v, params, x)

  return networks.FeedForwardNetwork(init=init_fn, apply=apply_fn)


class EnsembleTest(absltest.TestCase):

  def test_ensemble_init(self):
    x = jnp.ones(10)  # Base input

    wrapped_ffn = params_adding_ffn(x)

    rr_ensemble = ensemble.make_ensemble(
        wrapped_ffn, ensemble.apply_round_robin, num_networks=3)
    key = jax.random.PRNGKey(0)
    params = rr_ensemble.init(key)

    self.assertTupleEqual(params.shape, (3,) + x.shape)

    # The ensemble dimension is the lead dimension.
    self.assertFalse((params[0, ...] == params[1, ...]).all())

  def test_apply_all(self):
    x = jnp.ones(10)  # Base input
    bx = jnp.ones((7, 10))  # Batched input

    wrapped_ffn = params_adding_ffn(x)

    rr_ensemble = ensemble.make_ensemble(
        wrapped_ffn, ensemble.apply_all, num_networks=3)
    key = jax.random.PRNGKey(0)
    params = rr_ensemble.init(key)
    self.assertTupleEqual(params.shape, (3,) + x.shape)

    y = rr_ensemble.apply(params, x)
    self.assertTupleEqual(y.shape, (3,) + x.shape)
    np.testing.assert_allclose(params, y - jnp.broadcast_to(x, (3,) + x.shape))

    by = rr_ensemble.apply(params, bx)
    # Note: the batch dimension is no longer the leading dimension.
    self.assertTupleEqual(by.shape, (3,) + bx.shape)

  def test_apply_round_robin(self):
    x = jnp.ones(10)  # Base input
    bx = jnp.ones((7, 10))  # Batched input

    wrapped_ffn = params_adding_ffn(x)

    rr_ensemble = ensemble.make_ensemble(
        wrapped_ffn, ensemble.apply_round_robin, num_networks=3)
    key = jax.random.PRNGKey(0)
    params = rr_ensemble.init(key)
    self.assertTupleEqual(params.shape, (3,) + x.shape)

    y = rr_ensemble.apply(params, jnp.broadcast_to(x, (3,) + x.shape))
    self.assertTupleEqual(y.shape, (3,) + x.shape)
    np.testing.assert_allclose(params, y - x)

    # Note: the ensemble dimension must lead, the batch dimension is no longer
    # the leading dimension.
    by = rr_ensemble.apply(
        params, jnp.broadcast_to(jnp.expand_dims(bx, axis=0), (3,) + bx.shape))
    self.assertTupleEqual(by.shape, (3,) + bx.shape)

    # If num_networks=3, then `round_robin(params, input)[4]` should be equal
    # to `apply(params[1], input[4])`, etc.
    yy = rr_ensemble.apply(params, jnp.broadcast_to(x, (6,) + x.shape))
    self.assertTupleEqual(yy.shape, (6,) + x.shape)
    np.testing.assert_allclose(
        jnp.concatenate([params, params], axis=0),
        yy - jnp.expand_dims(x, axis=0))

  def test_apply_mean(self):
    x = jnp.ones(10)  # Base input
    bx = jnp.ones((7, 10))  # Batched input

    wrapped_ffn = params_adding_ffn(x)

    rr_ensemble = ensemble.make_ensemble(
        wrapped_ffn, ensemble.apply_mean, num_networks=3)
    key = jax.random.PRNGKey(0)
    params = rr_ensemble.init(key)
    self.assertTupleEqual(params.shape, (3,) + x.shape)
    self.assertFalse((params[0, ...] == params[1, ...]).all())

    y = rr_ensemble.apply(params, x)
    self.assertTupleEqual(y.shape, x.shape)
    np.testing.assert_allclose(
        jnp.mean(params, axis=0), y - x, atol=1E-5, rtol=1E-5)

    by = rr_ensemble.apply(params, bx)
    self.assertTupleEqual(by.shape, bx.shape)

  def test_apply_all_multiargs(self):
    x = jnp.ones(10)  # Base input

    wrapped_ffn = funny_args_ffn(x)

    rr_ensemble = ensemble.make_ensemble(
        wrapped_ffn, ensemble.apply_all, num_networks=3)
    key = jax.random.PRNGKey(0)
    params = rr_ensemble.init(key)
    self.assertTupleEqual(params.shape, (3,) + x.shape)

    y = rr_ensemble.apply(params, x, 2 * x, x)
    self.assertTupleEqual(y.shape, (3,) + x.shape)
    np.testing.assert_allclose(
        params,
        y - jnp.broadcast_to(2 * x, (3,) + x.shape),
        atol=1E-5,
        rtol=1E-5)

    y = rr_ensemble.apply(params, x, bar=x, foo=2 * x)
    self.assertTupleEqual(y.shape, (3,) + x.shape)
    np.testing.assert_allclose(
        params,
        y - jnp.broadcast_to(2 * x, (3,) + x.shape),
        atol=1E-5,
        rtol=1E-5)

  def test_apply_all_structured(self):
    x = jnp.ones(10)
    sx = [(3 * x, 2 * x), 5 * x]  # Base input

    wrapped_ffn = struct_params_adding_ffn(sx)

    rr_ensemble = ensemble.make_ensemble(
        wrapped_ffn, ensemble.apply_all, num_networks=3)
    key = jax.random.PRNGKey(0)
    params = rr_ensemble.init(key)

    y = rr_ensemble.apply(params, sx)
    ex = jnp.broadcast_to(x, (3,) + x.shape)
    np.testing.assert_allclose(y[0][0], params[0][0] + 3 * ex)

  def test_apply_round_robin_multiargs(self):
    x = jnp.ones(10)  # Base input

    wrapped_ffn = funny_args_ffn(x)

    rr_ensemble = ensemble.make_ensemble(
        wrapped_ffn, ensemble.apply_round_robin, num_networks=3)
    key = jax.random.PRNGKey(0)
    params = rr_ensemble.init(key)
    self.assertTupleEqual(params.shape, (3,) + x.shape)

    ex = jnp.broadcast_to(x, (3,) + x.shape)
    y = rr_ensemble.apply(params, ex, 2 * ex, ex)
    self.assertTupleEqual(y.shape, (3,) + x.shape)
    np.testing.assert_allclose(
        params,
        y - jnp.broadcast_to(2 * x, (3,) + x.shape),
        atol=1E-5,
        rtol=1E-5)

    y = rr_ensemble.apply(params, ex, bar=ex, foo=2 * ex)
    self.assertTupleEqual(y.shape, (3,) + x.shape)
    np.testing.assert_allclose(
        params,
        y - jnp.broadcast_to(2 * x, (3,) + x.shape),
        atol=1E-5,
        rtol=1E-5)

  def test_apply_round_robin_structured(self):
    x = jnp.ones(10)
    sx = [(3 * x, 2 * x), 5 * x]  # Base input

    wrapped_ffn = struct_params_adding_ffn(sx)

    rr_ensemble = ensemble.make_ensemble(
        wrapped_ffn, ensemble.apply_round_robin, num_networks=3)
    key = jax.random.PRNGKey(0)
    params = rr_ensemble.init(key)

    ex = jnp.broadcast_to(x, (3,) + x.shape)
    esx = [(3 * ex, 2 * ex), 5 * ex]
    y = rr_ensemble.apply(params, esx)
    np.testing.assert_allclose(y[0][0], params[0][0] + 3 * ex)

  def test_apply_mean_multiargs(self):
    x = jnp.ones(10)  # Base input

    wrapped_ffn = funny_args_ffn(x)

    rr_ensemble = ensemble.make_ensemble(
        wrapped_ffn, ensemble.apply_mean, num_networks=3)
    key = jax.random.PRNGKey(0)
    params = rr_ensemble.init(key)
    self.assertTupleEqual(params.shape, (3,) + x.shape)

    y = rr_ensemble.apply(params, x, 2 * x, x)
    self.assertTupleEqual(y.shape, x.shape)
    np.testing.assert_allclose(
        jnp.mean(params, axis=0), y - 2 * x, atol=1E-5, rtol=1E-5)

    y = rr_ensemble.apply(params, x, bar=x, foo=2 * x)
    self.assertTupleEqual(y.shape, x.shape)
    np.testing.assert_allclose(
        jnp.mean(params, axis=0), y - 2 * x, atol=1E-5, rtol=1E-5)

  def test_apply_mean_structured(self):
    x = jnp.ones(10)
    sx = [(3 * x, 2 * x), 5 * x]  # Base input

    wrapped_ffn = struct_params_adding_ffn(sx)

    rr_ensemble = ensemble.make_ensemble(
        wrapped_ffn, ensemble.apply_mean, num_networks=3)
    key = jax.random.PRNGKey(0)
    params = rr_ensemble.init(key)

    y = rr_ensemble.apply(params, sx)
    np.testing.assert_allclose(
        y[0][0], jnp.mean(params[0][0], axis=0) + 3 * x, atol=1E-5, rtol=1E-5)

  def test_round_robin_random(self):
    x = jnp.ones(10)  # Base input
    bx = jnp.ones((9, 10))  # Batched input
    ffn = RandomFFN()
    wrapped_ffn = networks.FeedForwardNetwork(
        init=functools.partial(ffn.init, x=x), apply=ffn.apply)
    rr_ensemble = ensemble.make_ensemble(
        wrapped_ffn, ensemble.apply_round_robin, num_networks=3)

    key = jax.random.PRNGKey(0)
    params = rr_ensemble.init(key)
    out = rr_ensemble.apply(params, bx)
    # The output should be the same every 3 rows:
    blocks = jnp.split(out, 3, axis=0)
    np.testing.assert_array_equal(blocks[0], blocks[1])
    np.testing.assert_array_equal(blocks[0], blocks[2])
    self.assertTrue((out[0] != out[1]).any())

    for i in range(9):
      np.testing.assert_allclose(
          out[i],
          ffn.apply(jax.tree_map(lambda p, i=i: p[i % 3], params), bx[i]),
          atol=1E-5,
          rtol=1E-5)

  def test_mean_random(self):
    x = jnp.ones(10)
    bx = jnp.ones((9, 10))
    ffn = RandomFFN()
    wrapped_ffn = networks.FeedForwardNetwork(
        init=functools.partial(ffn.init, x=x), apply=ffn.apply)
    mean_ensemble = ensemble.make_ensemble(
        wrapped_ffn, ensemble.apply_mean, num_networks=3)
    key = jax.random.PRNGKey(0)
    params = mean_ensemble.init(key)
    single_output = mean_ensemble.apply(params, x)
    self.assertEqual(single_output.shape, (15,))
    batch_output = mean_ensemble.apply(params, bx)
    # Make sure all rows are equal:
    np.testing.assert_allclose(
        jnp.broadcast_to(batch_output[0], batch_output.shape),
        batch_output,
        atol=1E-5,
        rtol=1E-5)

    # Check results explicitly:
    all_members = jnp.concatenate([
        jnp.expand_dims(
            ffn.apply(jax.tree_map(lambda p, i=i: p[i], params), bx), axis=0)
        for i in range(3)
    ])
    batch_means = jnp.mean(all_members, axis=0)
    np.testing.assert_allclose(batch_output, batch_means, atol=1E-5, rtol=1E-5)


if __name__ == '__main__':
  absltest.main()
