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

"""Tests for variable utilities."""

from acme.jax import variable_utils
from acme.testing import fakes
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tree

from absl.testing import absltest


def dummy_network(x):
  return hk.nets.MLP([50, 10])(x)


class VariableClientTest(absltest.TestCase):

  def test_update(self):
    init_fn, _ = hk.without_apply_rng(
        hk.transform(dummy_network))
    params = init_fn(jax.random.PRNGKey(1), jnp.zeros(shape=(1, 32)))
    variable_source = fakes.VariableSource(params)
    variable_client = variable_utils.VariableClient(
        variable_source, key='policy')
    variable_client.update_and_wait()
    tree.map_structure(np.testing.assert_array_equal, variable_client.params,
                       params)

  def test_multiple_keys(self):
    init_fn, _ = hk.without_apply_rng(
        hk.transform(dummy_network))
    params = init_fn(jax.random.PRNGKey(1), jnp.zeros(shape=(1, 32)))
    steps = jnp.zeros(shape=1)
    variables = {'network': params, 'steps': steps}
    variable_source = fakes.VariableSource(variables, use_default_key=False)
    variable_client = variable_utils.VariableClient(
        variable_source, key=['network', 'steps'])
    variable_client.update_and_wait()

    tree.map_structure(np.testing.assert_array_equal, variable_client.params[0],
                       params)
    tree.map_structure(np.testing.assert_array_equal, variable_client.params[1],
                       steps)


if __name__ == '__main__':
  absltest.main()
