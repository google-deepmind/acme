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

"""Behavior Value Estimation loss."""
import dataclasses
from typing import Tuple

from acme import types
from acme.agents.jax import dqn
from acme.jax import networks as networks_lib
import jax
import jax.numpy as jnp
import reverb
import rlax


@dataclasses.dataclass
class BVELoss(dqn.LossFn):
  """This loss implements TD-loss to estimate behavior value.

    This loss function uses the next action to learn with the SARSA tuples.
    It is intended to be used with dqn.SGDLearner. The method was proposed
    in "Regularized Behavior Value Estimation" by Gulcehre et al to overcome
    the extrapolation error in offline RL setting:
    https://arxiv.org/abs/2103.09575
  """
  discount: float = 0.99
  max_abs_reward: float = 1.
  huber_loss_parameter: float = 1.

  def __call__(
      self,
      network: networks_lib.TypedFeedForwardNetwork,
      params: networks_lib.Params,
      target_params: networks_lib.Params,
      batch: reverb.ReplaySample,
      key: networks_lib.PRNGKey,
  ) -> Tuple[jax.Array, dqn.LossExtra]:
    """Calculate a loss on a single batch of data."""
    transitions: types.Transition = batch.data

    # Forward pass.
    key1, key2 = jax.random.split(key)
    q_tm1 = network.apply(
        params, transitions.observation, is_training=True, key=key1)
    q_t_value = network.apply(
        target_params, transitions.next_observation, is_training=True, key=key2)

    # Cast and clip rewards.
    d_t = (transitions.discount * self.discount).astype(jnp.float32)
    r_t = jnp.clip(transitions.reward, -self.max_abs_reward,
                   self.max_abs_reward).astype(jnp.float32)

    # Compute double Q-learning n-step TD-error.
    batch_error = jax.vmap(rlax.sarsa)
    next_action = transitions.extras['next_action']
    td_error = batch_error(q_tm1, transitions.action, r_t, d_t, q_t_value,
                           next_action)
    batch_loss = rlax.huber_loss(td_error, self.huber_loss_parameter)

    # Average:
    loss = jnp.mean(batch_loss)  # []
    metrics = {'td_error': td_error, 'batch_loss': batch_loss}
    return loss, dqn.LossExtra(
        metrics=metrics,
        reverb_priorities=jnp.abs(td_error).astype(jnp.float64))
