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

"""ValueDice config."""

import dataclasses

from acme.adders import reverb as adders_reverb


@dataclasses.dataclass
class ValueDiceConfig:
  """Configuration options for ValueDice."""

  policy_learning_rate: float = 1e-5
  nu_learning_rate: float = 1e-3
  discount: float = .99
  batch_size: int = 256
  alpha: float = 0.05
  policy_reg_scale: float = 1e-4
  nu_reg_scale: float = 10.0

  # Replay options
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
  samples_per_insert: float = 256 * 4
  # Rate to be used for the SampleToInsertRatio rate limitter tolerance.
  # See a formula in make_replay_tables for more details.
  samples_per_insert_tolerance_rate: float = 0.1
  min_replay_size: int = 1000
  max_replay_size: int = 1000000
  prefetch_size: int = 4

  # How many gradient updates to perform per step.
  num_sgd_steps_per_step: int = 1
