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

"""TD3 config."""
import dataclasses
from typing import Optional, Union

from acme.adders import reverb as adders_reverb
import optax


@dataclasses.dataclass
class TD3Config:
  """Configuration options for TD3."""

  # Loss options
  batch_size: int = 256
  policy_learning_rate: Union[optax.Schedule, float] = 3e-4
  critic_learning_rate: Union[optax.Schedule, float] = 3e-4
  # Policy gradient clipping is not part of the original TD3 implementation,
  # used e.g. in DAC https://arxiv.org/pdf/1809.02925.pdf
  policy_gradient_clipping: Optional[float] = None
  discount: float = 0.99
  n_step: int = 1

  # TD3 specific options (https://arxiv.org/pdf/1802.09477.pdf)
  sigma: float = 0.1
  delay: int = 2
  target_sigma: float = 0.2
  noise_clip: float = 0.5
  tau: float = 0.005

  # Replay options
  min_replay_size: int = 1000
  max_replay_size: int = 1000000
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
  prefetch_size: int = 4
  samples_per_insert: float = 256
  # Rate to be used for the SampleToInsertRatio rate limiter tolerance.
  # See a formula in make_replay_tables for more details.
  samples_per_insert_tolerance_rate: float = 0.1

  # How many gradient updates to perform per step.
  num_sgd_steps_per_step: int = 1

  # Offline RL options
  # if bc_alpha: if given, will add a bc regularization term to the policy loss,
  # (https://arxiv.org/pdf/2106.06860.pdf), useful for offline training.
  bc_alpha: Optional[float] = None
