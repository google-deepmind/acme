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

"""Defines the available MPO configuration options."""

import dataclasses
from typing import Callable, Optional, Union

from acme import types
from acme.agents.jax.mpo import types as mpo_types
import numpy as np
import rlax


@dataclasses.dataclass
class MPOConfig:
  """MPO agent configuration."""

  batch_size: int = 256  # Total batch size across all learner devices.
  discount: float = 0.99
  discrete_policy: bool = False

  # Specification of the type of experience the learner will consume.
  experience_type: mpo_types.ExperienceType = dataclasses.field(
      default_factory=lambda: mpo_types.FromTransitions(n_step=5)
  )
  num_stacked_observations: int = 1
  # Optional data-augmentation transformation for observations.
  observation_transform: Optional[Callable[[types.NestedTensor],
                                           types.NestedTensor]] = None

  # Specification of replay, e.g., min/max size, pure or mixed.
  # NOTE: When replay_fraction = 1.0, this reverts to pure replay and the online
  # queue is not created.
  replay_fraction: float = 1.0  # Fraction of replay data (vs online) per batch.
  samples_per_insert: Optional[float] = 32.0
  min_replay_size: int = 1_000
  max_replay_size: int = 1_000_000
  online_queue_capacity: int = 0  # If not set, will use 4 * online_batch_size.

  # Critic training configuration.
  critic_type: mpo_types.CriticType = mpo_types.CriticType.MIXTURE_OF_GAUSSIANS
  value_tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR
  use_retrace: bool = False
  retrace_lambda: float = 0.95
  reward_clip: float = np.float32('inf')  # pytype: disable=annotation-type-mismatch  # numpy-scalars
  use_online_policy_to_bootstrap: bool = False
  use_stale_state: bool = False

  # Policy training configuration.
  num_samples: int = 20  # Number of MPO action samples.
  policy_loss_config: Optional[mpo_types.PolicyLossConfig] = None
  policy_eval_stochastic: bool = True
  policy_eval_num_val_samples: int = 128

  # Optimizer configuration.
  learning_rate: Union[float, Callable[[int], float]] = 1e-4
  dual_learning_rate: Union[float, Callable[[int], float]] = 1e-2
  grad_norm_clip: float = 40.
  adam_b1: float = 0.9
  adam_b2: float = 0.999
  weight_decay: float = 0.0
  use_cosine_lr_decay: bool = False
  cosine_lr_decay_warmup_steps: int = 3000

  # Set the target update period or rate depending on whether you want a
  # periodic or incremental (exponential weighted average) target update.
  # Exactly one must be specified (not None).
  target_update_period: Optional[int] = 100
  target_update_rate: Optional[float] = None
  variable_update_period: int = 1000

  # Configuring the mixture of policy and critic losses.
  policy_loss_scale: float = 1.0
  critic_loss_scale: float = 1.0

  # Optional roll-out loss configuration (off by default).
  model_rollout_length: int = 0
  rollout_policy_loss_scale: float = 1.0
  rollout_bc_policy_loss_scale: float = 1.0
  rollout_critic_loss_scale: float = 1.0
  rollout_reward_loss_scale: float = 1.0

  jit_learner: bool = True

  def __post_init__(self):
    if ((self.target_update_period and self.target_update_rate) or
        (self.target_update_period is None and
         self.target_update_rate is None)):
      raise ValueError(
          'Exactly one of target_update_{period|rate} must be set.'
          f' Received target_update_period={self.target_update_period} and'
          f' target_update_rate={self.target_update_rate}.')

    online_batch_size = int(self.batch_size * (1. - self.replay_fraction))
    if not self.online_queue_capacity:
      # Note: larger capacities mean the online data is more "stale". This seems
      # a reasonable default for now.
      self.online_queue_capacity = int(4 * online_batch_size)
    self.online_queue_capacity = max(self.online_queue_capacity,
                                     online_batch_size + 1)

    if self.samples_per_insert is not None and self.replay_fraction < 1:
      raise ValueError(
          'Cannot set samples_per_insert when using a mixed replay (i.e when '
          '0 < replay_fraction < 1). Received:\n'
          f'\tsamples_per_insert={self.samples_per_insert} and\n'
          f'\treplay_fraction={self.replay_fraction}.')

    if (0 < self.replay_fraction < 1 and
        self.min_replay_size > self.online_queue_capacity):
      raise ValueError('When mixing replay with an online queue, min replay '
                       'size must not be larger than the queue capacity.')

    if (isinstance(self.experience_type, mpo_types.FromTransitions) and
        self.num_stacked_observations > 1):
      raise ValueError(
          'Agent-side frame-stacking is currently only supported when learning '
          'from sequences. Consider environment-side frame-stacking instead.')

    if self.critic_type == mpo_types.CriticType.CATEGORICAL:
      if self.model_rollout_length > 0:
        raise ValueError(
            'Model rollouts are not supported for the Categorical critic')
      if not isinstance(self.experience_type, mpo_types.FromTransitions):
        raise ValueError(
            'Categorical critic only supports experience_type=FromTransitions')
      if self.use_retrace:
        raise ValueError('retrace is not supported for the Categorical critic')

    if self.model_rollout_length > 0 and not self.discrete_policy:
      if (self.rollout_policy_loss_scale or self.rollout_bc_policy_loss_scale):
        raise ValueError('Policy rollout losses are only supported in the '
                         'discrete policy case.')


def _compute_spi_from_replay_fraction(replay_fraction: float) -> float:
  """Computes an estimated samples_per_insert from a replay_fraction.

  Assumes actors simultaneously add to both the queue and replay in a mixed
  replay setup. Since the online queue sets samples_per_insert = 1, then the
  total SPI can be calculated as:

    SPI = B / O = O / (1 - f) / O = 1 / (1 - f).

  Key:
    B: total batch size
    O: online batch size
    f: replay fraction.

  Args:
    replay_fraction: fraction of a batch size taken from replay (as opposed to
      the queue of online experience) in a mixed replay setting.

  Returns:
    An estimate of the samples_per_insert value to produce comparable runs in
    the pure replay setting.
  """
  return 1 / (1 - replay_fraction)


def _compute_num_inserts_per_actor_step(samples_per_insert: float,
                                        batch_size: int,
                                        sequence_period: int = 1) -> float:
  """Estimate the number inserts per actor steps."""
  return sequence_period * batch_size / samples_per_insert
