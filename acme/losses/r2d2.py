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

"""Loss functions for R2D2."""

from typing import Iterable, NamedTuple, Sequence

import tensorflow as tf
import trfl


class LossCoreExtra(NamedTuple):
  targets: tf.Tensor
  errors: tf.Tensor


def transformed_n_step_loss(
    qs: tf.Tensor,
    targnet_qs: tf.Tensor,
    actions: tf.Tensor,
    rewards: tf.Tensor,
    pcontinues: tf.Tensor,
    target_policy_probs: tf.Tensor,
    bootstrap_n: int,
    stop_targnet_gradients: bool = True,
    name: str = 'transformed_n_step_loss',
) -> trfl.base_ops.LossOutput:
  """Helper function for computing transformed loss on sequences.

  Args:
    qs: 3-D tensor corresponding to the Q-values to be learned. Shape is [T+1,
      B, A].
    targnet_qs: Like `qs`, but in the target network setting, these values
      should be computed by the target network. Shape is [T+1, B, A].
    actions: 2-D tensor holding the indices of actions executed during the
      transition that corresponds to each major index. Shape is [T+1, B].
    rewards: 2-D tensor holding rewards received during the transition that
      corresponds to each major index. Shape is [T, B].
    pcontinues: 2-D tensor holding pcontinue values received during the
      transition that corresponds to each major index. Shape is [T, B].
    target_policy_probs: 3-D tensor holding per-action policy probabilities for
      the states encountered just before taking the transitions that correspond
      to each major index, according to the target policy (i.e. the policy we
      wish to learn). For standard Q-learning the probabilities should form a
      one-hot vector over actions where the nonzero index corresponds to the max
      Q. Shape is [T+1, B, A].
    bootstrap_n: Transition length for N-step bootstrapping.
    stop_targnet_gradients: `bool` indicating whether to apply tf.stop_gradients
      to the target values. This should usually be True.
    name: name to prefix ops created by this function.

  Returns:
    a tuple of:
    * `loss`: the transformed Q-learning loss summed over `T`.
    * `LossCoreExtra`: namedtuple containing the fields `targets` and `errors`.
  """

  with tf.name_scope(name):
    # Require correct tensor ranks---as long as we have shape information
    # available to check. If there isn't any, we print a warning.
    def check_rank(tensors: Iterable[tf.Tensor], ranks: Sequence[int]):
      for i, (tensor, rank) in enumerate(zip(tensors, ranks)):
        if tensor.get_shape():
          trfl.assert_rank_and_shape_compatibility([tensor], rank)
        else:
          raise ValueError(
              f'Tensor "{tensor.name}", which was offered as transformed_n_step_loss'
              f'parameter {i+1}, has no rank at construction time, so cannot verify'
              f'that it has the necessary rank of {rank}')

    check_rank(
        [qs, targnet_qs, actions, rewards, pcontinues, target_policy_probs],
        [3, 3, 2, 2, 2, 3])

    # Construct arguments to compute bootstrap target.
    a_tm1 = actions[:-1]  # (0:T) x B
    r_t, pcont_t = rewards, pcontinues  # (1:T+1) x B
    q_tm1 = qs[:-1]  # (0:T) x B x A
    target_policy_t = target_policy_probs[1:]  # (1:T+1) x B x A
    targnet_q_t = targnet_qs[1:]  # (1:T+1) x B x A

    bootstrap_value = tf.reduce_sum(
        target_policy_t * _signed_parabolic_tx(targnet_q_t), -1)
    target = _compute_n_step_sequence_targets(
        r_t=r_t,
        pcont_t=pcont_t,
        bootstrap_value=bootstrap_value,
        n=bootstrap_n)

    if stop_targnet_gradients:
      target = tf.stop_gradient(target)

    # tx/inv_tx may result in numerical instabilities so mask any NaNs.
    finite_mask = tf.math.is_finite(target)
    target = tf.where(finite_mask, target, tf.zeros_like(target))

    qa_tm1 = trfl.batched_index(q_tm1, a_tm1)
    errors = qa_tm1 - _signed_hyperbolic_tx(target)

    # Only compute n-step errors w.r.t. finite targets.
    errors = tf.where(finite_mask, errors, tf.zeros_like(errors))

    # Sum over time dimension.
    loss = 0.5 * tf.reduce_sum(tf.square(errors), axis=0)

    return trfl.base_ops.LossOutput(
        loss, LossCoreExtra(targets=target, errors=errors))


def _compute_n_step_sequence_targets(
    r_t: tf.Tensor,
    pcont_t: tf.Tensor,
    bootstrap_value: tf.Tensor,
    n: int,
) -> tf.Tensor:
  """Computes n-step bootstrapped returns over a sequence.

  Args:
    r_t: 2-D tensor of shape [T, B] corresponding to rewards.
    pcont_t: 2-D tensor of shape [T, B] corresponding to pcontinues.
    bootstrap_value: 2-D tensor of shape [T, B] corresponding to bootstrap
      values.
    n: number of steps over which to accumulate reward before bootstrapping.

  Returns:
    2-D tensor of shape [T, B] corresponding to bootstrapped returns.
  """
  time_size, batch_size = r_t.shape.as_list()

  # Pad r_t and pcont_t so we can use static slice shapes in scan.
  r_t = tf.concat([r_t, tf.zeros((n - 1, batch_size))], 0)
  pcont_t = tf.concat([pcont_t, tf.ones((n - 1, batch_size))], 0)

  # We need to use tf.slice with static shapes for TPU compatibility.
  def _slice(tensor, index, size):
    return tf.slice(tensor, [index, 0], [size, batch_size])

  # Construct correct bootstrap targets for each time slice t, which are exactly
  # the target values at timestep min(t+n-1, time_size-1).
  last_bootstrap_value = _slice(bootstrap_value, time_size - 1, 1)
  if time_size > n - 1:
    full_bootstrap_steps = [_slice(bootstrap_value, n - 1, time_size - (n - 1))]
    truncated_bootstrap_steps = [last_bootstrap_value] * (n - 1)
  else:
    # Only truncated steps, since n > time_size.
    full_bootstrap_steps = []
    truncated_bootstrap_steps = [last_bootstrap_value] * time_size
  bootstrap_value = tf.concat(full_bootstrap_steps + truncated_bootstrap_steps,
                              0)

  # Iterate backwards for n steps to construct n-step return targets.
  targets = bootstrap_value
  for i in range(n - 1, -1, -1):
    this_pcont_t = _slice(pcont_t, i, time_size)
    this_r_t = _slice(r_t, i, time_size)
    targets = this_r_t + this_pcont_t * targets
  return targets


def _signed_hyperbolic_tx(x: tf.Tensor, eps: float = 1e-3) -> tf.Tensor:
  """Signed hyperbolic transform, inverse of signed_parabolic."""
  return tf.sign(x) * (tf.sqrt(abs(x) + 1) - 1) + eps * x


def _signed_parabolic_tx(x: tf.Tensor, eps: float = 1e-3) -> tf.Tensor:
  """Signed parabolic transform, inverse of signed_hyperbolic."""
  z = tf.sqrt(1 + 4 * eps * (eps + 1 + abs(x))) / 2 / eps - 1 / 2 / eps
  return tf.sign(x) * (tf.square(z) - 1)
