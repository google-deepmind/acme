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

"""BVE config."""

import dataclasses
from typing import Callable, Sequence, Union

import numpy as np


@dataclasses.dataclass
class BVEConfig:
  """Configuration options for BVE agent.

  Attributes:
    epsilon: for use by epsilon-greedy policies. If multiple, the epsilons are
      alternated randomly per-episode.
    seed: Random seed.
    learning_rate: Learning rate for Adam optimizer. Could be a number or a
      function defining a schedule.
    adam_eps: Epsilon for Adam optimizer.
    discount: Discount rate applied to value per timestep.
    target_update_period: Update target network every period.
    max_gradient_norm: For gradient clipping.
    max_abs_reward: Maximum absolute reward.
    huber_loss_parameter: The delta parameter of the huber loss.
    batch_size: Number of transitions per batch.
    prefetch_size: Prefetch size for reverb replay performance.
    num_sgd_steps_per_step: How many gradient updates to perform per learner
      step.
  """
  epsilon: Union[float, Sequence[float]] = 0.05
  # TODO(b/191706065): update all clients and remove this field.
  seed: int = 1

  # Learning rule
  learning_rate: Union[float, Callable[[int], float]] = 3e-4
  adam_eps: float = 1e-8  # Eps for Adam optimizer.
  discount: float = 0.99  # Discount rate applied to value per timestep.
  target_update_period: int = 2500  # Update target network every period.
  max_gradient_norm: float = np.inf  # For gradient clipping.
  max_abs_reward: float = 1.  # Maximum absolute value to clip the rewards.
  huber_loss_parameter: float = 1.  # Huber loss delta parameter.
  batch_size: int = 256  # Minibatch size.
  prefetch_size = 500  # The amount of data to prefetch into the memory.
  num_sgd_steps_per_step: int = 1
