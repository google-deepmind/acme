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

"""DQN config."""

import dataclasses

from acme.adders import reverb as adders_reverb
import numpy as np


@dataclasses.dataclass
class DQNConfig:
  """Configuration options for DQN agent."""
  epsilon: float = 0.05  # Action selection via epsilon-greedy policy.
  # TODO(b/191706065): update all clients and remove this field.
  seed: int = 1  # Random seed.

  # Learning rule
  learning_rate: float = 1e-3  # Learning rate for Adam optimizer.
  discount: float = 0.99  # Discount rate applied to value per timestep.
  n_step: int = 5  # N-step TD learning.
  target_update_period: int = 100  # Update target network every period.
  max_gradient_norm: float = np.inf  # For gradient clipping.

  # Replay options
  batch_size: int = 256  # Number of transitions per batch.
  min_replay_size: int = 1_000  # Minimum replay size.
  max_replay_size: int = 1_000_000  # Maximum replay size.
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
  importance_sampling_exponent: float = 0.2  # Importance sampling for replay.
  priority_exponent: float = 0.6  # Priority exponent for replay.
  prefetch_size: int = 4  # Prefetch size for reverb replay performance.
  samples_per_insert: float = 0.5  # Ratio of learning samples to insert.
  # Rate to be used for the SampleToInsertRatio rate limitter tolerance.
  # See a formula in make_replay_tables for more details.
  samples_per_insert_tolerance_rate: float = 0.1

  # How many gradient updates to perform per learner step.
  num_sgd_steps_per_step: int = 1
