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

from acme.adders import reverb as adders_reverb
import rlax


@dataclasses.dataclass
class R2D2Config:
  """Configuration options for R2D2 agent."""
  discount: float = 0.997
  target_update_period: int = 2500
  evaluation_epsilon: float = 0.
  num_epsilons: int = 256
  variable_update_period: int = 400

  # Learner options
  burn_in_length: int = 40
  trace_length: int = 80
  sequence_period: int = 40
  learning_rate: float = 1e-3
  bootstrap_n: int = 5
  clip_rewards: bool = False
  tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR

  # Replay options
  samples_per_insert_tolerance_rate: float = 0.1
  samples_per_insert: float = 4.0
  min_replay_size: int = 50_000
  max_replay_size: int = 100_000
  batch_size: int = 64
  prefetch_size: int = 2
  num_parallel_calls: int = 16
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE

  # Priority options
  importance_sampling_exponent: float = 0.6
  priority_exponent: float = 0.9
  max_priority_weight: float = 0.9
