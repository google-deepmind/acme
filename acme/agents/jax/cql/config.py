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

"""Config classes for CQL."""
import dataclasses
from typing import Optional


@dataclasses.dataclass
class CQLConfig:
  """Configuration options for CQL.

  Attributes:
    batch_size: batch size.
    policy_learning_rate: learning rate for the policy optimizer.
    critic_learning_rate: learning rate for the Q-function optimizer.
    tau: Target smoothing coefficient.
    fixed_cql_coefficient: the value for cql coefficient. If None an adaptive
      coefficient will be used.
    cql_lagrange_threshold: a threshold that controls the adaptive loss for the
      cql coefficient.
    cql_num_samples: number of samples used to compute logsumexp(Q) via
      importance sampling.
    num_sgd_steps_per_step: how many gradient updates to perform per batch.
      Batch is split into this many smaller batches thus should be a multiple of
      num_sgd_steps_per_step
    reward_scale: reward scale.
    discount: discount to use for TD updates.
    fixed_entropy_coefficient: coefficient applied to the entropy bonus. If None
      an adaptative coefficient will be used.
    target_entropy: target entropy when using adapdative entropy bonus.
    num_bc_iters: number of BC steps for actor initialization.
  """
  batch_size: int = 256
  policy_learning_rate: float = 3e-5
  critic_learning_rate: float = 3e-4
  fixed_cql_coefficient: float = 5.
  tau: float = 0.005
  fixed_cql_coefficient: Optional[float] = 5.
  cql_lagrange_threshold: Optional[float] = None
  cql_num_samples: int = 10
  num_sgd_steps_per_step: int = 1
  reward_scale: float = 1.0
  discount: float = 0.99
  fixed_entropy_coefficient: Optional[float] = 0.
  target_entropy: Optional[float] = 0
  num_bc_iters: int = 50_000
