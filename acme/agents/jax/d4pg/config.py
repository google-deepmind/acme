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

"""Config classes for D4PG."""
import dataclasses
from typing import Optional
from acme.adders import reverb as adders_reverb


@dataclasses.dataclass
class D4PGConfig:
  """Configuration options for D4PG."""
  sigma: float = 0.3
  target_update_period: int = 100
  samples_per_insert: Optional[float] = 32.0

  # Loss options
  n_step: int = 5
  discount: float = 0.99
  batch_size: int = 256
  learning_rate: float = 1e-4
  clipping: bool = True

  # Replay options
  min_replay_size: int = 1000
  max_replay_size: int = 1000000
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
  prefetch_size: int = 4
  # Rate to be used for the SampleToInsertRatio rate limitter tolerance.
  # See a formula in make_replay_tables for more details.
  samples_per_insert_tolerance_rate: float = 0.1

  # How many gradient updates to perform per step.
  num_sgd_steps_per_step: int = 1
