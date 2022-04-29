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

"""Tests for acme.tf.networks.distributional."""

from acme.tf.networks import distributional
import numpy as np
from numpy import testing as npt

from absl.testing import absltest
from absl.testing import parameterized


class DistributionalTest(parameterized.TestCase):

  @parameterized.parameters(
      ((2, 3), (), (), 5, (2, 5)),
      ((2, 3), (4, 1), (1, 5), 6, (2, 4, 5, 6)),
      )
  def test_discrete_valued_head(
      self,
      input_shape,
      vmin_shape,
      vmax_shape,
      num_atoms,
      expected_logits_shape):

    vmin = np.zeros(vmin_shape, float)
    vmax = np.ones(vmax_shape, float)
    head = distributional.DiscreteValuedHead(
        vmin=vmin,
        vmax=vmax,
        num_atoms=num_atoms)
    input_array = np.zeros(input_shape, dtype=float)
    output_distribution = head(input_array)
    self.assertEqual(output_distribution.logits_parameter().shape,
                     expected_logits_shape)

    values = output_distribution._values

    # Can't do assert_allclose(values[..., 0], vmin), because the args may
    # have broadcast-compatible but unequal shapes. Do the following instead:
    npt.assert_allclose(values[..., 0] - vmin, np.zeros_like(values[..., 0]))
    npt.assert_allclose(values[..., -1] - vmax, np.zeros_like(values[..., -1]))

    # Check that values are monotonically increasing.
    intervals = values[..., 1:] - values[..., :-1]
    npt.assert_array_less(np.zeros_like(intervals), intervals)

    # Check that the values are equally spaced.
    npt.assert_allclose(intervals[..., 1:] - intervals[..., :1],
                        np.zeros_like(intervals[..., 1:]),
                        atol=1e-7)


if __name__ == '__main__':
  absltest.main()
