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

"""Loss function for IMPALA (Espeholt et al., 2018) [1].

[1] https://arxiv.org/abs/1802.01561
"""

import collections
from typing import Callable

from acme.jax import utils
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import reverb
import rlax
import tree


Array = chex.Array
VTraceOutput = collections.namedtuple(
    'vtrace_output', ['errors', 'pg_advantage', 'q_estimate'])


def vtrace(
    v_tm1: Array,
    v_t: Array,
    r_t: Array,
    discount_t: Array,
    rho_t: Array,
    lambda_: float = 1.0,
    clip_rho_threshold: float = 1.0,
    stop_target_gradients: bool = True,
) -> Array:
  """Calculates V-Trace errors from importance weights.

  V-trace computes TD-errors from multistep trajectories by applying
  off-policy corrections based on clipped importance sampling ratios.

  See "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor
  Learner Architectures" by Espeholt et al. (https://arxiv.org/abs/1802.01561).

  Args:
    v_tm1: values at time t-1.
    v_t: values at time t.
    r_t: reward at time t.
    discount_t: discount at time t.
    rho_t: importance sampling ratios.
    lambda_: scalar mixing parameter lambda.
    clip_rho_threshold: clip threshold for importance weights.
    stop_target_gradients: whether or not to apply stop gradient to targets.

  Returns:
    V-Trace error.
  """
  chex.assert_rank([v_tm1, v_t, r_t, discount_t, rho_t], [1, 1, 1, 1, 1])
  chex.assert_type([v_tm1, v_t, r_t, discount_t, rho_t],
                   [float, float, float, float, float])
  chex.assert_equal_shape([v_tm1, v_t, r_t, discount_t, rho_t])

  # Clip importance sampling ratios.
  c_t = jnp.minimum(1.0, rho_t) * lambda_
  clipped_rhos = jnp.minimum(clip_rho_threshold, rho_t)

  # Compute the temporal difference errors.
  td_errors = clipped_rhos * (r_t + discount_t * v_t - v_tm1)

  # Work backwards computing the td-errors.
  err = 0.0
  errors = []
  for i in jnp.arange(v_t.shape[0] - 1, -1, -1):
    err = td_errors[i] + discount_t[i] * c_t[i] * err
    errors.insert(0, err)

  # Return errors.
  if not stop_target_gradients:
    return jnp.array(errors)
  # In TD-like algorithms, we want gradients to only flow in the predictions,
  # and not in the values used to bootstrap. In this case, add the value of the
  # initial state value to get the implied estimates of the returns, stop
  # gradient around such target and then subtract again the initial state value.
  else:
    target_tm1 = jnp.array(errors) + v_tm1
    target_tm1 = jax.lax.stop_gradient(target_tm1)
  return target_tm1 - v_tm1


def vtrace_td_error_and_advantage(
    v_tm1: Array,
    v_t: Array,
    r_t: Array,
    discount_t: Array,
    rho_t: Array,
    lambda_: float = 1.0,
    clip_rho_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0,
    stop_target_gradients: bool = True,
) -> VTraceOutput:
  """Calculates V-Trace errors and PG advantage from importance weights.

  This functions computes the TD-errors and policy gradient Advantage terms
  as used by the IMPALA distributed actor-critic agent.

  See "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor
  Learner Architectures" by Espeholt et al. (https://arxiv.org/abs/1802.01561)

  Args:
    v_tm1: values at time t-1.
    v_t: values at time t.
    r_t: reward at time t.
    discount_t: discount at time t.
    rho_t: importance weights at time t.
    lambda_: scalar mixing parameter lambda.
    clip_rho_threshold: clip threshold for importance ratios.
    clip_pg_rho_threshold: clip threshold for policy gradient importance ratios.
    stop_target_gradients: whether or not to apply stop gradient to targets.

  Returns:
    a tuple of V-Trace error, policy gradient advantage, and estimated Q-values.
  """
  chex.assert_rank([v_tm1, v_t, r_t, discount_t, rho_t], 1)
  chex.assert_type([v_tm1, v_t, r_t, discount_t, rho_t], float)
  chex.assert_equal_shape([v_tm1, v_t, r_t, discount_t, rho_t])

  errors = vtrace(
      v_tm1, v_t, r_t, discount_t, rho_t,
      lambda_, clip_rho_threshold, stop_target_gradients)
  targets_tm1 = errors + v_tm1
  q_bootstrap = jnp.concatenate([
      lambda_ * targets_tm1[1:] + (1 - lambda_) * v_tm1[1:],
      v_t[-1:],
  ], axis=0)
  q_estimate = r_t + discount_t * q_bootstrap
  clipped_pg_rho_tm1 = jnp.minimum(clip_pg_rho_threshold, rho_t)
  pg_advantages = clipped_pg_rho_tm1 * (q_estimate - v_tm1)
  return VTraceOutput(
      errors=errors, pg_advantage=pg_advantages, q_estimate=q_estimate)


def impala_loss(
    unroll_fn: hk.Transformed,
    *,
    discount: float,
    max_abs_reward: float = np.inf,
    baseline_cost: float = 1.,
    entropy_cost: float = 0.,
) -> Callable[[hk.Params, reverb.ReplaySample], jnp.DeviceArray]:
  """Builds the standard entropy-regularised IMPALA loss function.

  Args:
    unroll_fn: A `hk.Transformed` object containing a callable which maps
      (params, observations_sequence, initial_state) -> ((logits, value), state)
    discount: The standard geometric discount rate to apply.
    max_abs_reward: Optional symmetric reward clipping to apply.
    baseline_cost: Weighting of the critic loss relative to the policy loss.
    entropy_cost: Weighting of the entropy regulariser relative to policy loss.

  Returns:
    A loss function with signature (params, data) -> loss_scalar.
  """

  def loss_fn(params: hk.Params,
              sample: reverb.ReplaySample) -> jnp.DeviceArray:
    """Batched, entropy-regularised actor-critic loss with V-trace."""

    # Extract the data.
    data = sample.data
    observations, actions, rewards, discounts, extra = (data.observation,
                                                        data.action,
                                                        data.reward,
                                                        data.discount,
                                                        data.extras)
    initial_state = tree.map_structure(lambda s: s[0], extra['core_state'])
    behaviour_logits = extra['logits']

    # Apply reward clipping.
    rewards = jnp.clip(rewards, -max_abs_reward, max_abs_reward)

    # Unroll current policy over observations.
    (logits, values), _ = unroll_fn.apply(params, observations, initial_state)

    # Compute importance sampling weights: current policy / behavior policy.
    rhos = rlax.categorical_importance_sampling_ratios(logits[:-1],
                                                       behaviour_logits[:-1],
                                                       actions[:-1])

    # Critic loss.
    vtrace_returns = vtrace_td_error_and_advantage(
        v_tm1=values[:-1],
        v_t=values[1:],
        r_t=rewards[:-1],
        discount_t=discounts[:-1] * discount,
        rho_t=rhos)
    critic_loss = jnp.square(vtrace_returns.errors)

    # Policy gradient loss.
    policy_gradient_loss = rlax.policy_gradient_loss(
        logits_t=logits[:-1],
        a_t=actions[:-1],
        adv_t=vtrace_returns.pg_advantage,
        w_t=jnp.ones_like(rewards[:-1]))

    # Entropy regulariser.
    entropy_loss = rlax.entropy_loss(logits[:-1], jnp.ones_like(rewards[:-1]))

    # Combine weighted sum of actor & critic losses, averaged over the sequence.
    mean_loss = jnp.mean(policy_gradient_loss + baseline_cost * critic_loss +
                         entropy_cost * entropy_loss)  # []

    return mean_loss

  return utils.mapreduce(loss_fn, in_axes=(None, 0))
