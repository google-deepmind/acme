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

"""DQN losses."""
from typing import Tuple

from acme.agents.jax.dqn import learning_lib
import dataclasses
import haiku as hk
import jax
import jax.numpy as jnp
import reverb
import rlax


@dataclasses.dataclass
class PrioritizedDoubleQLearning(learning_lib.LossFn):
  """Clipped double q learning with prioritization on TD error."""
  discount: float = 0.99
  importance_sampling_exponent: float = 0.2
  max_abs_reward: float = 1.
  huber_loss_parameter: float = 1.

  def __call__(
      self,
      network: hk.Transformed,
      params: hk.Params,
      target_params: hk.Params,
      batch: reverb.ReplaySample,
      key: jnp.DeviceArray,
  ) -> Tuple[jnp.DeviceArray, learning_lib.LossExtra]:
    """Calculate a loss on a single batch of data."""
    del key
    o_tm1, a_tm1, r_t, d_t, o_t = batch.data
    keys, probs, *_ = batch.info

    # Forward pass.
    q_tm1 = network.apply(params, o_tm1)
    q_t_value = network.apply(target_params, o_t)
    q_t_selector = network.apply(params, o_t)

    # Cast and clip rewards.
    d_t = (d_t * self.discount).astype(jnp.float32)
    r_t = jnp.clip(
        r_t, -self.max_abs_reward, self.max_abs_reward).astype(jnp.float32)

    # Compute double Q-learning n-step TD-error.
    batch_error = jax.vmap(rlax.double_q_learning)
    td_error = batch_error(q_tm1, a_tm1, r_t, d_t, q_t_value, q_t_selector)
    batch_loss = rlax.huber_loss(td_error, self.huber_loss_parameter)

    # Importance weighting.
    importance_weights = (1. / probs).astype(jnp.float32)
    importance_weights **= self.importance_sampling_exponent
    importance_weights /= jnp.max(importance_weights)

    # Reweight.
    loss = jnp.mean(importance_weights * batch_loss)  # []
    reverb_update = learning_lib.ReverbUpdate(
        keys=keys, priorities=jnp.abs(td_error).astype(jnp.float64))
    extra = learning_lib.LossExtra(metrics={}, reverb_update=reverb_update)
    return loss, extra

