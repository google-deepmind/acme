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

"""PWIL config."""
import dataclasses
from typing import Iterator

from acme import types


@dataclasses.dataclass
class PWILConfig:
  """Configuration options for PWIL.

  The default values correspond to the experiment setup from the PWIL
  publication http://arxiv.org/abs/2006.04678.
  """

  # Number of transitions to fill the replay buffer with for pretraining.
  num_transitions_rb: int = 50000

  # If False, uses only observations for computing the distance; if True, also
  # uses the actions.
  use_actions_for_distance: bool = True

  # Scaling for the reward function, see equation (6) in
  # http://arxiv.org/abs/2006.04678.
  alpha: float = 5.

  # Controls the kernel size of the reward function, see equation (6)
  # in http://arxiv.org/abs/2006.04678.
  beta: float = 5.

  # When False, uses the reward signal from the dataset during prefilling.
  prefill_constant_reward: bool = True

  num_sgd_steps_per_step: int = 1


@dataclasses.dataclass
class PWILDemonstrations:
  """Unbatched, unshuffled transitions with approximate episode length."""
  demonstrations: Iterator[types.Transition]
  episode_length: int
