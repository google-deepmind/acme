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

"""Configuration options for Implicit Q-Learning (IQL)."""
import dataclasses
from typing import Optional


@dataclasses.dataclass
class IQLConfig:
  """Configuration options for IQL.

  Attributes:
    batch_size: Batch size for training.
    value_learning_rate: Learning rate for the value function optimizer.
    critic_learning_rate: Learning rate for the Q-function optimizer.
    policy_learning_rate: Learning rate for the policy optimizer.
    tau: Target network update coefficient (Polyak averaging).
    expectile: Expectile parameter for value function (τ in paper).
      Higher values (e.g., 0.9) are more conservative.
    temperature: Inverse temperature (β) for advantage-weighted regression.
      Higher values give more weight to high-advantage actions.
    discount: Discount factor for TD updates.
    num_sgd_steps_per_step: Number of gradient updates per environment step.
    num_bc_iters: Number of behavioral cloning iterations for policy warmup.
  """
  batch_size: int = 256
  value_learning_rate: float = 3e-4
  critic_learning_rate: float = 3e-4
  policy_learning_rate: float = 3e-4
  tau: float = 0.005
  expectile: float = 0.7
  temperature: float = 3.0
  discount: float = 0.99
  num_sgd_steps_per_step: int = 1
  num_bc_iters: int = 0
