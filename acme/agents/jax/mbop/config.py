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

"""MBOP config."""

import dataclasses

from acme.agents.jax.mbop import mppi


@dataclasses.dataclass(frozen=True)
class MBOPConfig:
  """Configuration options for the MBOP agent.

  Attributes:
    mppi_config: Planner hyperparameters.
    learning_rate: Learning rate.
    num_networks: Number of networks in the ensembles.
    num_sgd_steps_per_step: How many gradient updates to perform per learner
      step.
  """
  mppi_config: mppi.MPPIConfig = dataclasses.field(
      default_factory=mppi.MPPIConfig
  )
  learning_rate: float = 3e-4
  num_networks: int = 5
  num_sgd_steps_per_step: int = 1
