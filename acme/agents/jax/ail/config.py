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

"""AIL config."""
import dataclasses
from typing import Optional

import optax


@dataclasses.dataclass
class AILConfig:
  """Configuration options for AIL.

  Attributes:
    direct_rl_batch_size: Batch size of a direct rl algorithm (measured in
      transitions).
    is_sequence_based: If True, a direct rl algorithm is using SequenceAdder
      data format. Otherwise the learner assumes that the direct rl algorithm is
      using NStepTransitionAdder.
    share_iterator: If True, AIL will use the same iterator for the
      discriminator network training as the direct rl algorithm.
    num_sgd_steps_per_step: Only used if 'share_iterator' is False. Denotes how
      many gradient updates perform per one learner step.
    discriminator_batch_size:  Batch size for training the discriminator.
    policy_variable_name: The name of the policy variable to retrieve direct_rl
      policy parameters.
    discriminator_optimizer: Optimizer for the discriminator. If not specified
      it is set to Adam with learning rate of 1e-5.
    replay_table_name: The name of the reverb replay table to use.
    prefetch_size: How many batches to prefetch
    discount: Discount to use for TD updates
    min_replay_size: Minimal size of replay buffer
    max_replay_size: Maximal size of replay buffer
    policy_to_expert_data_ratio: If not None, the direct RL learner will receive
      expert transitions in the given proportions.
      policy_to_expert_data_ratio + 1 must divide the direct RL batch size.
  """
  direct_rl_batch_size: int
  is_sequence_based: bool = False
  share_iterator: bool = True
  num_sgd_steps_per_step: int = 1
  discriminator_batch_size: int = 256
  policy_variable_name: Optional[str] = None
  discriminator_optimizer: Optional[optax.GradientTransformation] = None
  replay_table_name: str = 'ail_table'
  prefetch_size: int = 4
  discount: float = 0.99
  min_replay_size: int = 1000
  max_replay_size: int = int(1e6)
  policy_to_expert_data_ratio: Optional[int] = None

  def __post_init__(self):
    assert self.direct_rl_batch_size % self.discriminator_batch_size == 0


def get_per_learner_step_batch_size(config: AILConfig) -> int:
  """Returns how many transitions should be sampled per direct learner step."""
  # If the iterators are tied, the discriminator learning batch size has to
  # match the direct RL one.
  if config.share_iterator:
    assert (config.direct_rl_batch_size % config.discriminator_batch_size) == 0
    return config.direct_rl_batch_size
  # Otherwise each iteration of the discriminator will sample a batch which will
  # be split in num_sgd_steps_per_step batches, each of size
  # discriminator_batch_size.
  return config.discriminator_batch_size * config.num_sgd_steps_per_step
