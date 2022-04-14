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
from typing import Callable, Sequence, Union

from acme.adders import reverb as adders_reverb
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass
class DQNConfig:
  """Configuration options for DQN agent.

  Attributes:
    epsilon: for use by epsilon-greedy policies. If multiple, the epsilons are
      alternated randomly per-episode.
    seed: Random seed.
    learning_rate: Learning rate for Adam optimizer. Could be a number or a
      function defining a schedule.
    adam_eps: Epsilon for Adam optimizer.
    discount: Discount rate applied to value per timestep.
    n_step: N-step TD learning.
    target_update_period: Update target network every period.
    max_gradient_norm: For gradient clipping.
    batch_size: Number of transitions per batch.
    min_replay_size: Minimum replay size.
    max_replay_size: Maximum replay size.
    replay_table_name: Reverb table, defaults to DEFAULT_PRIORITY_TABLE.
    importance_sampling_exponent: Importance sampling for replay.
    priority_exponent: Priority exponent for replay.
    prefetch_size: Prefetch size for reverb replay performance.
    samples_per_insert: Ratio of learning samples to insert.
    samples_per_insert_tolerance_rate: Rate to be used for
      the SampleToInsertRatio rate limitter tolerance.
      See a formula in make_replay_tables for more details.
    num_sgd_steps_per_step: How many gradient updates to perform per learner
      step.
  """
  epsilon: Union[float, Sequence[float]] = 0.05
  # TODO(b/191706065): update all clients and remove this field.
  seed: int = 1

  # Learning rule
  learning_rate: Union[float, Callable[[int], float]] = 1e-3
  adam_eps: float = 1e-8  # Eps for Adam optimizer.
  discount: float = 0.99  # Discount rate applied to value per timestep.
  n_step: int = 5  # N-step TD learning.
  target_update_period: int = 100  # Update target network every period.
  max_gradient_norm: float = np.inf  # For gradient clipping.

  # Replay options
  batch_size: int = 256
  min_replay_size: int = 1_000
  max_replay_size: int = 1_000_000
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
  importance_sampling_exponent: float = 0.2
  priority_exponent: float = 0.6
  prefetch_size: int = 4
  samples_per_insert: float = 0.5
  samples_per_insert_tolerance_rate: float = 0.1

  num_sgd_steps_per_step: int = 1


def logspace_epsilons(num_epsilons: int, epsilon: float = 0.017
                      ) -> Sequence[float]:
  """`num_epsilons` of logspace-distributed values, with median `epsilon`."""
  if num_epsilons <= 1:
    return (epsilon,)
  return jnp.logspace(1, 8, num_epsilons, base=epsilon ** (2./9.))
