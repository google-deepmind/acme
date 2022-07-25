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

"""Config classes for CRR."""
import dataclasses


@dataclasses.dataclass
class CRRConfig:
  """Configuration options for CRR.

  Attributes:
    learning_rate: Learning rate.
    discount: discount to use for TD updates.
    target_update_period: period to update target's parameters.
    use_sarsa_target: compute on-policy target using iterator's actions rather
      than sampled actions.
      Useful for 1-step offline RL (https://arxiv.org/pdf/2106.08909.pdf).
  """
  learning_rate: float = 3e-4
  discount: float = 0.99
  target_update_period: int = 100
  use_sarsa_target: bool = False
