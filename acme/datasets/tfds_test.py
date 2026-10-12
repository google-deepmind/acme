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

"""Tests for in-memory sampling of TFDS datasets."""

import inspect
from unittest import mock

from absl.testing import absltest
from acme.datasets import tfds
import jax
import numpy as np


class _FakeDataset:

  def __init__(self):
    self._data = np.arange(24, dtype=np.float32).reshape(8, 3)

  def cardinality(self):
    return len(self._data)

  def batch(self, unused_size):
    return self

  def as_numpy_iterator(self):
    return iter([self._data])


class InMemoryRandomSampleIteratorTest(absltest.TestCase):

  def test_unsharded_jitted_sampler_accepts_dataset_as_argument(self):
    jitted_functions = []
    original_jit = jax.jit

    def record_jit(fn):
      jitted_functions.append(fn)
      return original_jit(fn)

    with mock.patch.object(
        tfds.jax_utils, '_pmap_device_order',
        return_value=jax.devices()[:1]):
      with mock.patch.object(tfds.jax, 'jit', side_effect=record_jit):
        iterator = tfds.JaxInMemoryRandomSampleIterator(
            dataset=_FakeDataset(),
            key=jax.random.PRNGKey(0),
            batch_size=2)

    # The compiled sampler accepts dataset and key separately. This prevents
    # dataset values from being compiled as XLA constants.
    self.assertLen(jitted_functions, 1)
    self.assertEqual(
        list(inspect.signature(jitted_functions[0]).parameters),
        ['data', 'key'])

    self.assertEqual(iterator.dataset_size, 8)
    batch = next(iterator)
    self.assertEqual(batch.shape, (2, 3))
    self.assertTrue(np.isin(np.asarray(batch), np.arange(24)).all())


if __name__ == '__main__':
  absltest.main()
