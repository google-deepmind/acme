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

"""Builder for GAIL.

https://arxiv.org/pdf/1606.03476.pdf
"""

import dataclasses
from typing import Callable, Iterator

from acme import types
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import ppo
from acme.agents.jax.ail import builder
from acme.agents.jax.ail import config as ail_config
from acme.agents.jax.ail import losses


@dataclasses.dataclass
class GAILConfig:
  """Configuration options specific to GAIL."""
  ail_config: ail_config.AILConfig
  ppo_config: ppo.PPOConfig


class GAILBuilder(builder.AILBuilder[ppo.PPONetworks,
                                     actor_core_lib.FeedForwardPolicyWithExtra]
                 ):
  """GAIL Builder."""

  def __init__(self, config: GAILConfig,
               make_demonstrations: Callable[[int],
                                             Iterator[types.Transition]]):

    ppo_builder = ppo.PPOBuilder(config.ppo_config)
    super().__init__(
        ppo_builder,
        config=config.ail_config,
        discriminator_loss=losses.gail_loss(),
        make_demonstrations=make_demonstrations)
