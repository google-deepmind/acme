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

"""PPO config."""
import dataclasses
from typing import Callable, Union, Optional

from acme.adders import reverb as adders_reverb


@dataclasses.dataclass
class PPOConfig:
  """Configuration options for PPO.

  Attributes:
    unroll_length: Length of sequences added to the replay buffer.
    num_minibatches: The number of minibatches to split an epoch into.
      i.e. minibatch size = batch_size * unroll_length / num_minibatches.
    num_epochs: How many times to loop over the set of minibatches.
    batch_size: Number of trajectory segments of length unroll_length to gather
      for use in a call to the learner's step function.
    replay_table_name: Replay table name.
    ppo_clipping_epsilon: PPO clipping epsilon.
    normalize_advantage: Whether to normalize the advantages in the batch.
    normalize_value: Whether the critic should predict normalized values.
    normalization_ema_tau: Float tau for the exponential moving average used to
      maintain statistics for normalizing advantages and values.
    clip_value: Whether to clip the values as described in "What Matters in
      On-Policy Reinforcement Learning?".
    value_clipping_epsilon: Epsilon for value clipping.
    max_abs_reward: If provided clips the rewards in the trajectory to have
      absolute value less than or equal to max_abs_reward.
    gae_lambda: Lambda parameter in Generalized Advantage Estimation.
    discount: Discount factor.
    learning_rate: Learning rate for updating the policy and critic networks.
    adam_epsilon: Adam epsilon parameter.
    entropy_cost: Weight of the entropy regularizer term in policy optimization.
    value_cost: Weight of the value loss term in optimization.
    max_gradient_norm: Threshold for clipping the gradient norm.
    variable_update_period: Determines how frequently actors pull the parameters
      from the learner.
    log_global_norm_metrics: Whether to log global norm of gradients and
      updates.
    metrics_logging_period: How often metrics should be aggregated to host and
      logged.
  """
  unroll_length: int = 8
  num_minibatches: int = 8
  num_epochs: int = 2
  batch_size: int = 256
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
  ppo_clipping_epsilon: float = 0.2
  normalize_advantage: bool = False
  normalize_value: bool = False
  normalization_ema_tau: float = 0.995
  clip_value: bool = False
  value_clipping_epsilon: float = 0.2
  max_abs_reward: Optional[float] = None
  gae_lambda: float = 0.95
  discount: float = 0.99
  learning_rate: Union[float, Callable[[int], float]] = 3e-4
  adam_epsilon: float = 1e-7
  entropy_cost: float = 3e-4
  value_cost: float = 1.
  max_gradient_norm: float = 0.5
  variable_update_period: int = 1
  log_global_norm_metrics: bool = False
  metrics_logging_period: int = 100
