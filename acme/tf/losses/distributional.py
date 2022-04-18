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

from acme.tf import networks
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


def multiaxis_categorical(  # pylint: disable=invalid-name
    q_tm1: networks.DiscreteValuedDistribution,
    r_t: tf.Tensor,
    d_t: tf.Tensor,
    q_t: networks.DiscreteValuedDistribution) -> tf.Tensor:
  """Implements a multi-axis categorical distributional TD(0)-learning loss.

  All arguments may have a leading batch axis, but q_tm1.logits, and one of
  r_t or d_t *must* have a leading batch axis.

  Args:
    q_tm1: Previous timestep's value distribution.
    r_t: Reward.
    d_t: Discount.
    q_t: Current timestep's value distribution.

  Returns:
    Cross-entropy Bellman loss between q_tm1 and q_t + r_t * d_t.
    Shape: (B, *E), where
      B is the batch size.
      E is the broadcasted shape of r_t, d_t, and q_t.values[:-1].
  """
  tf.assert_equal(tf.rank(r_t), tf.rank(d_t))

  # Append a singleton axis corresponding to the axis that indexes the atoms in
  # q_t.values.
  r_t = r_t[..., None]  # shape: (B, *R, 1)
  d_t = d_t[..., None]  # shape: (B, *D, 1)

  z_t = r_t + d_t * q_t.values  # shape: (B, *E, N)

  p_t = tf.nn.softmax(q_t.logits)

  # Performs L2 projection.
  target = tf.stop_gradient(multiaxis_l2_project(z_t, p_t, q_t.values))

  # Calculates loss.
  loss = tf.nn.softmax_cross_entropy_with_logits(
      logits=q_tm1.logits, labels=target)

  return loss


# A modification of l2_project that allows multi-axis support arguments.
def multiaxis_l2_project(  # pylint: disable=invalid-name
    Zp: tf.Tensor,
    P: tf.Tensor,
    Zq: tf.Tensor,
) -> tf.Tensor:
  """Project distribution (Zp, P) onto support Zq under the L2-metric over CDFs.

  Let source support Zp's shape be described as (B, *C, M), where:
    B is the batch size.
    C contains the sizes of any axes in between the first and last axes.
    M is the number of atoms in the support.

  Let destination support Zq's shape be described as (*D, N), where:
    D contains the sizes of any axes before the last axis.
    N is the number of atoms in the support.

  Shapes C and D must have the same number of dimensions, and must be
  broadcastable with each other.

  Args:
    Zp: Support of source distribution. Shape: (B, *C, M).
    P:  Probability values of source distribution p(Zp[i]). Shape: (B, *C, M).
    Zq: Support to project P onto. Shape: (*D, N).

  Returns:
    The L2 projection of P from support Zp to support Zq.
    Shape: (B, *E, N), where E is the broadcast-merged shape of C and D.
  """

  tf.assert_equal(tf.shape(Zp), tf.shape(P))

  # Shapes C, D, and E as defined in the docstring above.
  shape_c = tf.shape(Zp)[1:-1]  # drop the batch and atom axes
  shape_d = tf.shape(Zq)[:-1]  # drop the atom axis
  shape_e = tf.broadcast_dynamic_shape(shape_c, shape_d)

  # If Zq has fewer inner axes than the broadcasted output shape, insert some
  # size-1 axes to broadcast.
  ndim_c = tf.size(shape_c)
  ndim_e = tf.size(shape_e)
  Zp = tf.reshape(
      Zp,
      tf.concat([tf.shape(Zp)[:1],  # B
                 tf.ones(tf.math.maximum(ndim_e - ndim_c, 0), dtype=tf.int32),
                 shape_c,  # C
                 tf.shape(Zp)[-1:]],  # M
                axis=0))
  P = tf.reshape(P, tf.shape(Zp))

  # Broadcast Zp, P, and Zq's common axes to the same shape: E.
  #
  # Normally it'd be sufficient to ensure that these args have the same number
  # of axes, then let the arithmetic operators broadcast as necessary. Instead,
  # we need to explicitly broadcast them here, because there's a call to
  # tf.clip_by_value(t, vmin, vmax) below, which doesn't allow t's dimensions
  # to be expanded to match vmin and vmax.

  # Shape: (B, *E, M)
  Zp = tf.broadcast_to(
      Zp,
      tf.concat([tf.shape(Zp)[:1],  # B
                 shape_e,  # E
                 tf.shape(Zp)[-1:]],  # M
                axis=0))

  # Shape: (B, *E, M)
  P = tf.broadcast_to(P, tf.shape(Zp))

  # Shape: (*E, N)
  Zq = tf.broadcast_to(Zq, tf.concat([shape_e, tf.shape(Zq)[-1:]], axis=0))

  # Extracts vmin and vmax and construct helper tensors from Zq.
  # These have shape shape_q, except the last axis has size 1.
  # Shape: (*E, 1)
  vmin, vmax = Zq[..., :1], Zq[..., -1:]

  # The distances between neighboring atom values in the target support.
  # Shape: (*E, N)
  d_pos = tf.roll(Zq, shift=-1, axis=-1) - Zq  # d_pos[i] := Zq[i+1] - Zq[i]
  d_neg = Zq - tf.roll(Zq, shift=1, axis=-1)   # d_neg[i] := Zq[i] - Zq[i-1]

  # Clips Zp to be in new support range (vmin, vmax).
  # Shape: (B, *E, 1, M)
  clipped_zp = tf.clip_by_value(Zp, vmin, vmax)[..., None, :]

  # Shape: (1, *E, N, 1)
  clipped_zq = Zq[None, ..., :, None]

  # Shape: (B, *E, N, M)
  delta_qp = clipped_zp - clipped_zq  # Zp[j] - Zq[i]

  # Shape: (B, *E, N, M)
  d_sign = tf.cast(delta_qp >= 0., dtype=P.dtype)

  # Insert singleton axes to d_pos and d_neg to maintain the same shape as
  # clipped_zq.
  # Shape: (1, *E, N, 1)
  d_pos = d_pos[None, ..., :, None]
  d_neg = d_neg[None, ..., :, None]

  # Shape: (B, *E, N, M)
  delta_hat = (d_sign * delta_qp / d_pos) - ((1. - d_sign) * delta_qp / d_neg)

  # Shape: (B, *E, 1, M)
  P = P[..., None, :]

  # Shape: (B, *E, N)
  return tf.reduce_sum(tf.clip_by_value(1. - delta_hat, 0., 1.) * P, axis=-1)
