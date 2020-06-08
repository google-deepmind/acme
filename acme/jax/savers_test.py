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

"""Tests for savers."""

from unittest import mock

from absl.testing import absltest
from acme import core
from acme.jax import savers
from acme.testing import test_utils
from acme.utils import paths
import jax.numpy as jnp
import numpy as np
import tree


class DummySaveable(core.Saveable):

  def __init__(self, state):
    self.state = state

  def save(self):
    return self.state

  def restore(self, state):
    self.state = state


def nest_assert_equal(a, b):
  tree.map_structure(np.testing.assert_array_equal, a, b)


class SaverTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    self._test_state = {
        'foo': jnp.ones(shape=(8, 4), dtype=jnp.float32),
        'bar': [jnp.zeros(shape=(3, 2), dtype=jnp.int32)],
        'baz': 3,
    }

  def test_save_restore(self):
    """Checks that we can save and restore state."""
    directory = self.get_tempdir()
    savers.save_to_path(directory, self._test_state)
    result = savers.restore_from_path(directory)
    nest_assert_equal(result, self._test_state)

  def test_checkpointer(self):
    """Checks that the Checkpointer class saves and restores as expected."""

    with mock.patch.object(paths, 'get_unique_id') as mock_unique_id:
      mock_unique_id.return_value = ('test',)

      # Given a path and some stateful object...
      directory = self.get_tempdir()
      x = DummySaveable(self._test_state)

      # If we checkpoint it...
      checkpointer = savers.Checkpointer(x, directory, time_delta_minutes=0)
      checkpointer.save()

      # The checkpointer should restore the object's state.
      x.state = None
      checkpointer.restore()
      nest_assert_equal(x.state, self._test_state)

      # Checkpointers should also attempt a restore at construction time.
      x.state = None
      savers.Checkpointer(x, directory, time_delta_minutes=0)
      nest_assert_equal(x.state, self._test_state)


if __name__ == '__main__':
  absltest.main()
