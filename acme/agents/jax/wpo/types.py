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

"""Some types/assumptions used in the WPO agent."""

import dataclasses
import enum
from typing import Callable, Mapping, Union

from acme import types
import acme.jax.losses.wpo as continuous_losses
import distrax
import jax.numpy as jnp


# TODO(bshahr): consider upstreaming these to core types.
NestedArray = types.NestedArray
Observation = types.NestedArray
ObservationEmbedding = types.NestedArray
Action = jnp.ndarray
RNGKey = jnp.ndarray
Entropy = jnp.ndarray
LogProb = jnp.ndarray
ExperienceType = Union['FromTransitions', 'FromSequences']

DistributionLike = distrax.DistributionLike
DistributionOrArray = DistributionLike | jnp.ndarray

LogDict = Mapping[str, jnp.ndarray]
PolicyStats = continuous_losses.WPOStats
DualParams = continuous_losses.WPOParams

DistributionalLossFn = Callable[[DistributionLike, jnp.ndarray], jnp.ndarray]


@dataclasses.dataclass
class FromTransitions:
  """Configuration for learning from n-step transitions."""
  n_step: int = 1
  # TODO(bshahr): consider adding the discount here.


@dataclasses.dataclass
class FromSequences:
  """Configuration for learning from sequences."""
  sequence_length: int = 10
  sequence_period: int = 1
  # Configuration of how to bootstrap from these sequences.
  n_step: int | None = 5
  # Lambda used to discount future rewards as in TD(lambda), Retrace, etc.
  td_lambda: float | None = 1.0


class RnnCoreType(enum.Enum):
  """Types of core that are supported for rnn."""
  IDENTITY = 'identity'
  GRU = 'gru'


@dataclasses.dataclass
class GaussianPolicyLossConfig:
  """Configuration for the continuous (Gaussian) policy loss."""
  epsilon_mean: float = 0.0025
  epsilon_stddev: float = 1e-6
  init_log_alpha_mean: float = 0.
  init_log_alpha_stddev: float = 10_000.
  policy_loss_scale: float = 1.0
  kl_loss_scale: float = 1.0
  # By default, disable the dual loss. This keeps the alpha parameters constant
  # and is equivalent to a soft KL loss rather than a hard KL loss.
  dual_loss_scale: float = 0.0
  per_dim_constraining: bool = True
  squashing_type: continuous_losses.WPOSquashingType = (
      continuous_losses.WPOSquashingType.IDENTITY)


PolicyLossConfig = GaussianPolicyLossConfig


@dataclasses.dataclass(frozen=True)
class LossScalesConfig:
  """Configuration for scaling the rollout losses used in the learner."""
  policy: float = 1.0
  critic: float = 1.0


@dataclasses.dataclass(frozen=True)
class ModelOutputs:
  """Container for the outputs of the model."""
  policy: types.NestedArray | None = None
  q_value: types.NestedArray | None = None
  value: types.NestedArray | None = None
  reward: types.NestedArray | None = None
  embedding: types.NestedArray | None = None
  recurrent_state: types.NestedArray | None = None
  actions: types.NestedArray | None = None
  q_values_grad: types.NestedArray | None = None


@dataclasses.dataclass(frozen=True)
class LossTargets:
  """Container for the targets used to compute the model loss."""
  # Policy targets.
  policy: types.NestedArray | None = None
  a_improvement: types.NestedArray | None = None
  q_improvement: types.NestedArray | None = None

  # Value targets.
  q_value: types.NestedArray | None = None
  value: types.NestedArray | None = None
  reward: types.NestedArray | None = None

  embedding: types.NestedArray | None = None
