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
from typing import Callable, Union

from acme.adders import reverb as adders_reverb
import numpy as np


@dataclasses.dataclass
class PPOConfig:
  """Configuration options for PPO."""
  unroll_length: int
  num_minibatches: int
  num_epochs: int

  batch_size: int = 1
  clip_value: bool = False
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
  ppo_clipping_epsilon: float = 0.2
  gae_lambda: float = 0.95
  discount: float = 0.99
  learning_rate: Union[float, Callable[[int], float]] = 1e-3
  adam_epsilon: float = 1e-5
  entropy_cost: float = 0.01
  value_cost: float = 1.
  max_abs_reward: float = np.inf
  max_gradient_norm: float = 0.5
  prefetch_size: int = 4
  variable_update_period: int = 1
