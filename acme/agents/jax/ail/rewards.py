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

"""AIL logits to AIL reward."""
from typing import Optional

from acme.agents.jax.ail import networks as ail_networks
from acme.jax import networks as networks_lib
import jax
import jax.numpy as jnp


def fairl_reward(
    max_reward_magnitude: Optional[float] = None
) -> ail_networks.ImitationRewardFn:
  """The FAIRL reward function (https://arxiv.org/pdf/1911.02256.pdf).

  Args:
    max_reward_magnitude: Clipping value for the reward.

  Returns:
    The function from logit to imitation reward.
  """

  def imitation_reward(logits: networks_lib.Logits) -> float:
    rewards = jnp.exp(jnp.clip(logits, a_max=20.)) * -logits
    if max_reward_magnitude is not None:
      # pylint: disable=invalid-unary-operand-type
      rewards = jnp.clip(
          rewards, a_min=-max_reward_magnitude, a_max=max_reward_magnitude)
    return rewards

  return imitation_reward


def gail_reward(
    reward_balance: float = .5,
    max_reward_magnitude: Optional[float] = None
) -> ail_networks.ImitationRewardFn:
  """GAIL reward function (https://arxiv.org/pdf/1606.03476.pdf).

  Args:
    reward_balance: 1 means log(D) reward, 0 means -log(1-D) and other values
      mean an average of the two.
    max_reward_magnitude: Clipping value for the reward.

  Returns:
    The function from logit to imitation reward.
  """

  def imitation_reward(logits: networks_lib.Logits) -> float:
    # Quick Maths:
    # logits = ln(D) - ln(1-D)
    # -softplus(-logits) = ln(D)
    # softplus(logits) = -ln(1-D)
    rewards = (
        reward_balance * -jax.nn.softplus(-logits) +
        (1 - reward_balance) * jax.nn.softplus(logits))
    if max_reward_magnitude is not None:
      # pylint: disable=invalid-unary-operand-type
      rewards = jnp.clip(
          rewards, a_min=-max_reward_magnitude, a_max=max_reward_magnitude)
    return rewards

  return imitation_reward
