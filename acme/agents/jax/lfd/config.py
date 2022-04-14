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

"""LfD config."""

import dataclasses


@dataclasses.dataclass
class LfdConfig:
  """Configuration options for LfD.

  Attributes:
    initial_insert_count: Number of steps of demonstrations to add to the replay
      buffer before adding any step of the collected episodes. Note that since
      only full episodes can be added, this number of steps is only a target.
    demonstration_ratio: Ratio of demonstration steps to add to the replay
      buffer. ratio = num_demonstration_steps_added / total_num_steps_added.
      The ratio must be in [0, 1).
      Note that this ratio is the desired ratio in the steady behavior and does
      not account for the initial demonstrations inserts.
      Note also that this ratio is only a target ratio since the granularity
      is the episode.
  """
  initial_insert_count: int = 0
  demonstration_ratio: float = 0.01
