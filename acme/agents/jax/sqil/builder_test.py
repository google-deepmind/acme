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

"""Tests for the SQIL iterator."""

from acme import types
from acme.agents.jax.sqil import builder
import numpy as np
import reverb

from absl.testing import absltest


class BuilderTest(absltest.TestCase):

  def test_sqil_iterator(self):
    demonstrations = [
        types.Transition(np.array([[1], [2], [3]]), (), (), (), ())
    ]
    replay = [
        reverb.ReplaySample(
            info=(),
            data=types.Transition(np.array([[4], [5], [6]]), (), (), (), ()))
    ]
    sqil_it = builder._generate_sqil_samples(iter(demonstrations), iter(replay))
    np.testing.assert_array_equal(
        next(sqil_it).data.observation, np.array([[1], [3], [5]]))
    np.testing.assert_array_equal(
        next(sqil_it).data.observation, np.array([[2], [4], [6]]))
    self.assertRaises(StopIteration, lambda: next(sqil_it))

if __name__ == '__main__':
  absltest.main()
