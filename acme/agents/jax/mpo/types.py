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

"""Some types/assumptions used in the MoG-MPO agent."""

import dataclasses
import enum
from typing import Callable, Mapping, Optional, Union

from acme import types
from acme.agents.jax.mpo import categorical_mpo as discrete_losses
import acme.jax.losses.mpo as continuous_losses
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
DistributionOrArray = Union[DistributionLike, jnp.ndarray]

LogDict = Mapping[str, jnp.ndarray]
PolicyStats = Union[
    discrete_losses.CategoricalMPOStats, continuous_losses.MPOStats]
DualParams = Union[continuous_losses.MPOParams,
                   discrete_losses.CategoricalMPOParams]

DistributionalLossFn = Callable[[DistributionLike, jnp.ndarray], jnp.ndarray]


@dataclasses.dataclass
class FromTransitions:
  """Configuration for learning from n-step transitions."""
  n_step: int = 1
  # TODO(bshahr): consider adding the discount here.


@dataclasses.dataclass
class FromSequences:
  """Configuration for learning from sequences."""
  sequence_length: int = 2
  sequence_period: int = 1
  # Configuration of how to bootstrap from these sequences.
  n_step: Optional[int] = 5
  # Lambda used to discount future rewards as in TD(lambda), Retrace, etc.
  td_lambda: Optional[float] = 1.0


class CriticType(enum.Enum):
  """Types of critic that are supported."""
  NONDISTRIBUTIONAL = 'nondistributional'
  MIXTURE_OF_GAUSSIANS = 'mixture_of_gaussians'
  CATEGORICAL_2HOT = 'categorical_2hot'
  CATEGORICAL = 'categorical'


class RnnCoreType(enum.Enum):
  """Types of core that are supported for rnn."""
  IDENTITY = 'identity'
  GRU = 'gru'


@dataclasses.dataclass
class GaussianPolicyLossConfig:
  """Configuration for the continuous (Gaussian) policy loss."""
  epsilon: float = 0.1
  epsilon_penalty: float = 0.001
  epsilon_mean: float = 0.0025
  epsilon_stddev: float = 1e-6
  init_log_temperature: float = 10.
  init_log_alpha_mean: float = 10.
  init_log_alpha_stddev: float = 1000.
  action_penalization: bool = True
  per_dim_constraining: bool = True


@dataclasses.dataclass
class CategoricalPolicyLossConfig:
  """Configuration for the discrete (categorical) policy loss."""
  epsilon: float = 0.1
  epsilon_policy: float = 0.0025
  init_log_temperature: float = 3.
  init_log_alpha: float = 3.


PolicyLossConfig = Union[GaussianPolicyLossConfig, CategoricalPolicyLossConfig]


@dataclasses.dataclass(frozen=True)
class RolloutLossScalesConfig:
  """Configuration for scaling the rollout losses used in the learner."""
  policy: float = 1.0
  bc_policy: float = 1.0
  critic: float = 1.0
  reward: float = 1.0


@dataclasses.dataclass(frozen=True)
class LossScalesConfig:
  """Configuration for scaling the rollout losses used in the learner."""
  policy: float = 1.0
  critic: float = 1.0
  rollout: Optional[RolloutLossScalesConfig] = None


@dataclasses.dataclass(frozen=True)
class ModelOutputs:
  """Container for the outputs of the model."""
  policy: Optional[types.NestedArray] = None
  q_value: Optional[types.NestedArray] = None
  value: Optional[types.NestedArray] = None
  reward: Optional[types.NestedArray] = None
  embedding: Optional[types.NestedArray] = None
  recurrent_state: Optional[types.NestedArray] = None


@dataclasses.dataclass(frozen=True)
class LossTargets:
  """Container for the targets used to compute the model loss."""
  # Policy targets.
  policy: Optional[types.NestedArray] = None
  a_improvement: Optional[types.NestedArray] = None
  q_improvement: Optional[types.NestedArray] = None

  # Value targets.
  q_value: Optional[types.NestedArray] = None
  value: Optional[types.NestedArray] = None
  reward: Optional[types.NestedArray] = None

  embedding: Optional[types.NestedArray] = None
