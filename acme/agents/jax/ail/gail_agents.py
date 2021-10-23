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

"""Defines distributed and local GAIL agents, using JAX.

https://arxiv.org/pdf/1606.03476.pdf
"""

import dataclasses
import functools
from typing import Callable

from acme import specs
from acme.agents.jax import ppo
from acme.agents.jax.ail import agents
from acme.agents.jax.ail import config as ail_config
from acme.agents.jax.ail import losses
from acme.agents.jax.ail import networks as ail_networks
from acme.jax import utils
from acme.utils import loggers
import dm_env


@dataclasses.dataclass
class GAILConfig:
  """Configuration options specific to GAIL."""
  ail_config: ail_config.AILConfig
  ppo_config: ppo.PPOConfig


class DistributedGAIL(agents.DistributedAIL):
  """Distributed program definition for GAIL."""

  def __init__(self,
               environment_factory: Callable[[bool], dm_env.Environment],
               config: GAILConfig,
               *args, **kwargs):
    logger_fn = functools.partial(
        loggers.make_default_logger,
        'direct_learner',
        kwargs['log_to_bigtable'],
        time_delta=kwargs['log_every'],
        asynchronous=True,
        serialize_fn=utils.fetch_devicearray,
        steps_key='learner_steps')
    ppo_agent = ppo.PPOBuilder(
        config.ppo_config, logger_fn=logger_fn)
    kwargs['discriminator_loss'] = losses.gail_loss()
    super().__init__(environment_factory, ppo_agent, config.ail_config, *args,
                     **kwargs)


class GAIL(agents.AIL):
  """Local agent for GAIL."""

  def __init__(self, spec: specs.EnvironmentSpec,
               network: ail_networks.AILNetworks, config: GAILConfig, *args,
               **kwargs):
    ppo_agent = ppo.PPOBuilder(config.ppo_config)
    kwargs['discriminator_loss'] = losses.gail_loss()
    super().__init__(spec, ppo_agent, network, config.ail_config, *args,
                     **kwargs)
