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

"""IMPALA config."""
import dataclasses
from typing import Optional, Union

from acme import types
from acme.adders import reverb as adders_reverb
import numpy as np


@dataclasses.dataclass
class IMPALAConfig:
  """Configuration options for IMPALA."""
  seed: int = 0

  # Loss options
  batch_size: int = 16
  prefetch_size: int = 2
  sequence_length: int = 20
  sequence_period: Optional[int] = None
  learning_rate: float = 1e-3
  adam_momentum_decay: float = 0.0
  adam_variance_decay: float = 0.99
  discount: float = 0.99
  entropy_cost: float = 0.01
  baseline_cost: float = 0.5
  max_abs_reward: float = np.inf
  max_gradient_norm: float = np.inf

  # Replay options
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
  num_prefetch_threads: Optional[int] = None
  samples_per_insert: Optional[float] = None
  max_queue_size: Union[int, types.Batches] = 10_000

  def __post_init__(self):
    if isinstance(self.max_queue_size, types.Batches):
      self.max_queue_size *= self.batch_size
    assert self.max_queue_size > self.batch_size + 1, ("""
        max_queue_size must be strictly larger than the batch size:
        - during the last step in an episode we might write 2 sequences to
          Reverb at once (that's how SequenceAdder works)
        - Reverb does insertion/sampling in multiple threads, so data is
          added asynchronously at unpredictable times. Therefore we need
          additional buffer size in order to avoid deadlocks.""")
