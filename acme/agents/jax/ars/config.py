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

"""ARS config."""
import dataclasses

from acme.adders import reverb as adders_reverb


@dataclasses.dataclass
class ARSConfig:
  """Configuration options for ARS."""
  num_steps: int = 1000000
  normalize_observations: bool = True
  step_size: float = 0.015
  num_directions: int = 60
  exploration_noise_std: float = 0.025
  top_directions: int = 20
  reward_shift: float = 1.0
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
