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

from typing import Callable

from acme.agents.jax.impala import types
from acme.jax import utils
import haiku as hk
import jax.numpy as jnp
import numpy as np
import reverb
import rlax
import tree


def impala_loss(
    unroll_fn: types.PolicyValueFn,
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
    (logits, values), _ = unroll_fn(params, observations, initial_state)

    # Compute importance sampling weights: current policy / behavior policy.
    rhos = rlax.categorical_importance_sampling_ratios(logits[:-1],
                                                       behaviour_logits[:-1],
                                                       actions[:-1])

    # Critic loss.
    vtrace_returns = rlax.vtrace_td_error_and_advantage(
        v_tm1=values[:-1],
        v_t=values[1:],
        r_t=rewards[:-1],
        discount_t=discounts[:-1] * discount,
        rho_tm1=rhos)
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

    metrics = {
        'policy_loss': jnp.mean(policy_gradient_loss),
        'critic_loss': jnp.mean(baseline_cost * critic_loss),
        'entropy_loss': jnp.mean(entropy_cost * entropy_loss),
        'entropy': jnp.mean(entropy_loss),
    }

    return mean_loss, metrics

  return utils.mapreduce(loss_fn, in_axes=(None, 0))
