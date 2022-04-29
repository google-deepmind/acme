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

"""Tests for the builder generator."""
from acme import types
from acme.agents.jax.ail import builder
import numpy as np
import reverb

from absl.testing import absltest

_REWARD = np.zeros((3,))


class BuilderTest(absltest.TestCase):

  def test_weighted_generator(self):
    data0 = types.Transition(np.array([[1], [2], [3]]), (), _REWARD, (), ())
    it0 = iter([data0])

    data1 = types.Transition(np.array([[4], [5], [6]]), (), _REWARD, (), ())
    data2 = types.Transition(np.array([[7], [8], [9]]), (), _REWARD, (), ())
    it1 = iter([
        reverb.ReplaySample(
            info=reverb.SampleInfo(
                *[() for _ in reverb.SampleInfo.tf_dtypes()]),
            data=data1),
        reverb.ReplaySample(
            info=reverb.SampleInfo(
                *[() for _ in reverb.SampleInfo.tf_dtypes()]),
            data=data2)
    ])

    weighted_it = builder._generate_samples_with_demonstrations(
        it0, it1, policy_to_expert_data_ratio=2, batch_size=3)

    np.testing.assert_array_equal(
        next(weighted_it).data.observation, np.array([[1], [4], [5]]))
    np.testing.assert_array_equal(
        next(weighted_it).data.observation, np.array([[7], [8], [2]]))
    self.assertRaises(StopIteration, lambda: next(weighted_it))


if __name__ == '__main__':
  absltest.main()
