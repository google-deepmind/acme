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

"""Builder for DAC.

https://arxiv.org/pdf/1809.02925.pdf
"""

import dataclasses
from typing import Callable, Iterator

from acme import types
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import td3
from acme.agents.jax.ail import builder
from acme.agents.jax.ail import config as ail_config
from acme.agents.jax.ail import losses


@dataclasses.dataclass
class DACConfig:
  """Configuration options specific to DAC.

  Attributes:
    ail_config: AIL config.
    td3_config: TD3 config.
    entropy_coefficient: Entropy coefficient of the discriminator loss.
    gradient_penalty_coefficient: Coefficient for the gradient penalty term in
      the discriminator loss.
  """
  ail_config: ail_config.AILConfig
  td3_config: td3.TD3Config
  entropy_coefficient: float = 1e-3
  gradient_penalty_coefficient: float = 10.


class DACBuilder(builder.AILBuilder[td3.TD3Networks,
                                    actor_core_lib.FeedForwardPolicy]):
  """DAC Builder."""

  def __init__(self, config: DACConfig,
               make_demonstrations: Callable[[int],
                                             Iterator[types.Transition]]):

    td3_builder = td3.TD3Builder(config.td3_config)
    dac_loss = losses.add_gradient_penalty(
        losses.gail_loss(entropy_coefficient=config.entropy_coefficient),
        gradient_penalty_coefficient=config.gradient_penalty_coefficient,
        gradient_penalty_target=1.)
    super().__init__(
        td3_builder,
        config=config.ail_config,
        discriminator_loss=dac_loss,
        make_demonstrations=make_demonstrations)
