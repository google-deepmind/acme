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

"""Tests for variable utilities."""

from absl.testing import absltest
from acme.testing import fakes
from acme.utils import jax_variable_utils

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tree


def dummy_network(x):
  return hk.nets.MLP([50, 10])(x)


class VariableClientTest(absltest.TestCase):

  def test_update(self):
    init_fn, _ = hk.transform(dummy_network)
    params = init_fn(jax.random.PRNGKey(1), jnp.zeros(shape=(1, 32)))
    variable_source = fakes.VariableSource(params)
    variable_client = jax_variable_utils.VariableClient(
        variable_source, key='policy')
    variable_client.update_and_wait()
    tree.map_structure(np.testing.assert_array_equal, variable_client.params,
                       params)


if __name__ == '__main__':
  absltest.main()
