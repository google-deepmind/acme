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

"""Losses and projection operators relevant to distributional RL."""

from acme import networks
import tensorflow as tf


def categorical(q_tm1: networks.DiscreteValuedDistribution, r_t: tf.Tensor,
                d_t: tf.Tensor,
                q_t: networks.DiscreteValuedDistribution) -> tf.Tensor:
  """Implements the Categorical Distributional TD(0)-learning loss."""

  z_t = tf.reshape(r_t, (-1, 1)) + tf.reshape(d_t, (-1, 1)) * q_t.values
  p_t = tf.nn.softmax(q_t.logits)

  # Performs L2 projection.
  target = tf.stop_gradient(l2_project(z_t, p_t, q_t.values))

  # Calculates loss.
  loss = tf.nn.softmax_cross_entropy_with_logits(
      logits=q_tm1.logits, labels=target)

  return loss


# Use an old version of the l2 projection which is probably slower on CPU
# but will run on GPUs.
def l2_project(  # pylint: disable=invalid-name
    Zp: tf.Tensor,
    P: tf.Tensor,
    Zq: tf.Tensor,
) -> tf.Tensor:
  """Project distribution (Zp, P) onto support Zq under the L2-metric over CDFs.

  This projection works for any support Zq.
  Let Kq be len(Zq) and Kp be len(Zp).

  Args:
    Zp: (batch_size, Kp) Support of distribution P
    P:  (batch_size, Kp) Probability values for P(Zp[i])
    Zq: (Kp,) Support to project onto

  Returns:
    L2 projection of (Zp, P) onto Zq.
  """

  # Asserts that Zq has no leading dimension of size 1.
  if Zq.get_shape().ndims > 1:
    Zq = tf.squeeze(Zq, axis=0)

  # Extracts vmin and vmax and construct helper tensors from Zq.
  vmin, vmax = Zq[0], Zq[-1]
  d_pos = tf.concat([Zq, vmin[None]], 0)[1:]
  d_neg = tf.concat([vmax[None], Zq], 0)[:-1]

  # Clips Zp to be in new support range (vmin, vmax).
  clipped_zp = tf.clip_by_value(Zp, vmin, vmax)[:, None, :]
  clipped_zq = Zq[None, :, None]

  # Gets the distance between atom values in support.
  d_pos = (d_pos - Zq)[None, :, None]  # Zq[i+1] - Zq[i]
  d_neg = (Zq - d_neg)[None, :, None]  # Zq[i] - Zq[i-1]

  delta_qp = clipped_zp - clipped_zq  # Zp[j] - Zq[i]

  d_sign = tf.cast(delta_qp >= 0., dtype=P.dtype)
  delta_hat = (d_sign * delta_qp / d_pos) - ((1. - d_sign) * delta_qp / d_neg)
  P = P[:, None, :]
  return tf.reduce_sum(tf.clip_by_value(1. - delta_hat, 0., 1.) * P, 2)
