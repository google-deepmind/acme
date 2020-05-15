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

"""Losses for Deterministic Policy Gradients."""

import tensorflow as tf


def dpg(
    q_max: tf.Tensor,
    a_max: tf.Tensor,
    tape: tf.GradientTape,
    dqda_clipping: float = None,
    clip_norm: bool = False,
) -> tf.Tensor:
  """Deterministic policy gradient loss, similar to trfl.dpg."""

  # Calculate the gradient dq/da.
  dqda = tape.gradient([q_max], [a_max])[0]

  if dqda is None:
    raise ValueError('q_max needs to be a function of a_max.')

  # Clipping the gradient dq/da.
  if dqda_clipping is not None:
    if dqda_clipping <= 0:
      raise ValueError('dqda_clipping should be bigger than 0, {} found'.format(
          dqda_clipping))
    if clip_norm:
      dqda = tf.clip_by_norm(dqda, dqda_clipping, axes=-1)
    else:
      dqda = tf.clip_by_value(dqda, -1. * dqda_clipping, dqda_clipping)

  # Target_a ensures correct gradient calculated during backprop.
  target_a = dqda + a_max
  # Stop the gradient going through Q network when backprop.
  target_a = tf.stop_gradient(target_a)
  # Gradient only go through actor network.
  loss = 0.5 * tf.reduce_sum(tf.square(target_a - a_max), axis=-1)
  # This recovers the DPG because (letting w be the actor network weights):
  # d(loss)/dw = 0.5 * (2 * (target_a - a_max) * d(target_a - a_max)/dw)
  #            = (target_a - a_max) * [d(target_a)/dw  - d(a_max)/dw]
  #            = dq/da * [d(target_a)/dw  - d(a_max)/dw]  # by defn of target_a
  #            = dq/da * [0 - d(a_max)/dw]                # by stop_gradient
  #            = - dq/da * da/dw

  return loss
