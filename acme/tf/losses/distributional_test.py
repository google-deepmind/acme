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

"""Tests for acme.tf.losses.distributional."""

from absl.testing import absltest
from absl.testing import parameterized
from acme.tf.losses import distributional
import numpy as np
from numpy import testing as npt
import tensorflow as tf


def _reference_l2_project(src_support, src_probs, dst_support):
  """Multi-axis l2_project, implemented using single-axis l2_project.

  This is for testing multiaxis_l2_project's consistency with l2_project,
  when used with multi-axis support vs single-axis support.

  Args:
    src_support: Zp in l2_project.
    src_probs: P in l2_project.
    dst_support: Zq in l2_project.

  Returns:
    src_probs, projected onto dst_support.
  """
  assert src_support.shape == src_probs.shape

  # Remove the batch and value axes, and broadcast the rest to a common shape.
  common_shape = np.broadcast(src_support[0, ..., 0],
                              dst_support[..., 0]).shape

  # If src_* have fewer internal axes than len(common_shape), insert size-1
  # axes.
  while src_support.ndim-2 < len(common_shape):
    src_support = src_support[:, None, ...]

  src_probs = np.reshape(src_probs, src_support.shape)

  # Broadcast args' non-batch, non-value axes to common_shape.
  src_support = np.broadcast_to(
      src_support,
      src_support.shape[:1] + common_shape + src_support.shape[-1:])
  src_probs = np.broadcast_to(src_probs, src_support.shape)
  dst_support = np.broadcast_to(
      dst_support,
      common_shape + dst_support.shape[-1:])

  output_shape = (src_support.shape[0],) + dst_support.shape

  # Collapse all but the first (batch) and last (atom) axes.
  src_support = src_support.reshape(
      [src_support.shape[0], -1, src_support.shape[-1]])
  src_probs = src_probs.reshape(
      [src_probs.shape[0], -1, src_probs.shape[-1]])

  # Collapse all but the last (atom) axes.
  dst_support = dst_support.reshape([-1, dst_support.shape[-1]])

  dst_probs = np.zeros(src_support.shape[:1] + dst_support.shape,
                       dtype=src_probs.dtype)

  # iterate over all supports
  for i in range(src_support.shape[1]):
    s_support = tf.convert_to_tensor(src_support[:, i, :])
    s_probs = tf.convert_to_tensor(src_probs[:, i, :])
    d_support = tf.convert_to_tensor(dst_support[i, :])
    d_probs = distributional.l2_project(s_support, s_probs, d_support)
    dst_probs[:, i, :] = d_probs.numpy()

  return dst_probs.reshape(output_shape)


class L2ProjectTest(parameterized.TestCase):

  @parameterized.parameters(
      [(2, 11), (11,)],  # C = (), D = (), matching num_atoms (11 and 11)
      [(2, 11), (5,)],  # C = (), D = (), differing num_atoms (11 and 5).
      [(2, 3, 11), (3, 5)],  # C = (3,), D = (3,)
      [(2, 1, 11), (3, 5)],  # C = (1,), D = (3,)
      [(2, 3, 11), (1, 5)],  # (C = (3,), D = (1,)
      [(2, 3, 4, 11), (3, 4, 5)],  # C = (3, 4), D = (3, 4)
      [(2, 3, 4, 11), (4, 5)],  # C = (3, 4), D = (4,)
      [(2, 4, 11), (3, 4, 5)],  # C = (4,), D = (3, 4)
  )
  def test_multiaxis(self, src_shape, dst_shape):
    """Tests consistency between multi-axis and single-axis l2_project.

    This calls l2_project on multi-axis supports, and checks that it gets
    the same outcomes as many calls to single-axis supports.

    Args:
      src_shape: Shape of source support. Includes a leading batch axis.
      dst_shape: Shape of destination support.
        Does not include a leading batch axis.
    """
    # src_shape includes a leading batch axis, whereas dst_shape does not.
    # assert len(src_shape) >= (1 + len(dst_shape))

    def make_support(shape, minimum):
      """Creates a ndarray of supports."""
      values = np.linspace(start=minimum, stop=minimum+100, num=shape[-1])
      offsets = np.arange(np.prod(shape[:-1]))
      result = values[None, :] + offsets[:, None]
      return result.reshape(shape)

    src_support = make_support(src_shape, -1)
    dst_support = make_support(dst_shape, -.75)

    rng = np.random.RandomState(1)
    src_probs = rng.uniform(low=1.0, high=2.0, size=src_shape)
    src_probs /= src_probs.sum()

    # Repeated calls to l2_project using single-axis supports.
    expected_dst_probs = _reference_l2_project(src_support,
                                               src_probs,
                                               dst_support)

    # A single call to l2_project, with multi-axis supports.
    dst_probs = distributional.multiaxis_l2_project(
        tf.convert_to_tensor(src_support),
        tf.convert_to_tensor(src_probs),
        tf.convert_to_tensor(dst_support)).numpy()

    npt.assert_allclose(dst_probs, expected_dst_probs)

  @parameterized.parameters(
      # Same src and dst support shape, dst support is shifted by +.25
      ([[0., 1, 2, 3]],
       [[0., 1, 0, 0]],
       [.25, 1.25, 2.25, 3.25],
       [[.25, .75, 0, 0]]),
      # Similar to above, but with batched src.
      ([[0., 1, 2, 3],
        [0., 1, 2, 3]],
       [[0., 1, 0, 0],
        [0., 0, 1, 0]],
       [.25, 1.25, 2.25, 3.25],
       [[.25, .75, 0, 0],
        [0., .25, .75, 0]]),
      # Similar to above, but src_probs has two 0.5's, instead of being one-hot.
      ([[0., 1, 2, 3]],
       [[0., .5, .5, 0]],
       [.25, 1.25, 2.25, 3.25],
       0.5 * (np.array([[.25, .75, 0, 0]]) + np.array([[0., .25, .75, 0]]))),
      # src and dst support have differing sizes
      ([[0., 1, 2, 3]],
       [[0., 1, 0, 0]],
       [0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50],
       [[0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]]),
      )
  def test_l2_projection(
      self, src_support, src_probs, dst_support, expected_dst_probs):

    dst_probs = distributional.multiaxis_l2_project(
        tf.convert_to_tensor(src_support),
        tf.convert_to_tensor(src_probs),
        tf.convert_to_tensor(dst_support)).numpy()
    npt.assert_allclose(dst_probs, expected_dst_probs)


if __name__ == '__main__':
  absltest.main()

