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

"""Tests for acme.tf.networks.distributions."""

from acme.tf.networks import distributions
import numpy as np
from numpy import testing as npt

from absl.testing import absltest
from absl.testing import parameterized


class DiscreteValuedDistributionTest(parameterized.TestCase):

  @parameterized.parameters(
      ((), (), 5),
      ((2,), (), 5),
      ((), (3, 4), 5),
      ((2,), (3, 4), 5),
      ((2, 6), (3, 4), 5),
      )
  def test_constructor(self, batch_shape, event_shape, num_values):
    logits_shape = batch_shape + event_shape + (num_values,)
    logits_size = np.prod(logits_shape)
    logits = np.arange(logits_size, dtype=float).reshape(logits_shape)
    values = np.linspace(start=-np.ones(event_shape, dtype=float),
                         stop=np.ones(event_shape, dtype=float),
                         num=num_values,
                         axis=-1)
    distribution = distributions.DiscreteValuedDistribution(values=values,
                                                            logits=logits)

    # Check batch and event shapes.
    self.assertEqual(distribution.batch_shape, batch_shape)
    self.assertEqual(distribution.event_shape, event_shape)
    self.assertEqual(distribution.logits_parameter().shape.as_list(),
                     list(logits.shape))
    self.assertEqual(distribution.logits_parameter().shape.as_list()[-1],
                     logits.shape[-1])

    # Test slicing
    if len(batch_shape) == 1:
      slice_0_logits = distribution[1:3].logits_parameter().numpy()
      expected_slice_0_logits = distribution.logits_parameter().numpy()[1:3]
      npt.assert_allclose(slice_0_logits, expected_slice_0_logits)
    elif len(batch_shape) == 2:
      slice_logits = distribution[0, 1:3].logits_parameter().numpy()
      expected_slice_logits = distribution.logits_parameter().numpy()[0, 1:3]
      npt.assert_allclose(slice_logits, expected_slice_logits)
    else:
      assert not batch_shape


if __name__ == '__main__':
  absltest.main()
