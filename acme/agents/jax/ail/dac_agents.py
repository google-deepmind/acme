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

"""Defines distributed and local DAC agents, using JAX.

https://arxiv.org/pdf/1809.02925.pdf
NOTE: The original DAC implementation uses TRPO but we use TD3 here.
"""

import dataclasses
import functools
from typing import Callable

from acme import specs
from acme.agents.jax import td3
from acme.agents.jax.ail import agents
from acme.agents.jax.ail import config as ail_config
from acme.agents.jax.ail import losses
from acme.agents.jax.ail import networks as ail_networks
from acme.jax import utils
from acme.utils import loggers
import dm_env


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


class DistributedDAC(agents.DistributedAIL):
  """Distributed program definition for DAC."""

  def __init__(self,
               environment_factory: Callable[[bool], dm_env.Environment],
               config: DACConfig,
               *args, **kwargs):
    logger_fn = functools.partial(
        loggers.make_default_logger,
        'direct_learner',
        kwargs['log_to_bigtable'],
        time_delta=kwargs['log_every'],
        asynchronous=True,
        serialize_fn=utils.fetch_devicearray,
        steps_key='learner_steps')
    td3_agent = td3.TD3Builder(config.td3_config, logger_fn=logger_fn)

    dac_loss = losses.add_gradient_penalty(
        losses.gail_loss(entropy_coefficient=config.entropy_coefficient),
        gradient_penalty_coefficient=config.gradient_penalty_coefficient,
        gradient_penalty_target=1.)
    kwargs['discriminator_loss'] = dac_loss
    super().__init__(environment_factory, td3_agent, config.ail_config, *args,
                     **kwargs)


class DAC(agents.AIL):
  """Local agent for DAC."""

  def __init__(self, spec: specs.EnvironmentSpec,
               network: ail_networks.AILNetworks, config: DACConfig, *args,
               **kwargs):
    td3_agent = td3.TD3Builder(config.td3_config)

    dac_loss = losses.add_gradient_penalty(
        losses.gail_loss(entropy_coefficient=config.entropy_coefficient),
        gradient_penalty_coefficient=config.gradient_penalty_coefficient,
        gradient_penalty_target=1.)
    kwargs['discriminator_loss'] = dac_loss
    super().__init__(spec, td3_agent, network, config.ail_config, *args,
                     **kwargs)
