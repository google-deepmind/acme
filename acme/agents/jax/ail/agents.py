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

"""Defines distributed and local AIL agents, using JAX."""

import functools
from typing import Any, Callable, Iterator, Optional

from acme import specs
from acme import types
from acme.agents.jax import builders
from acme.agents.jax.ail import builder
from acme.agents.jax.ail import config as ail_config
from acme.agents.jax.ail import losses
from acme.agents.jax.ail import networks as ail_networks
from acme.jax import utils
from acme.jax.layouts import distributed_layout
from acme.jax.layouts import local_layout
from acme.utils import counting
from acme.utils import loggers
import dm_env


NetworkFactory = Callable[[specs.EnvironmentSpec], ail_networks.AILNetworks]


class DistributedAIL(distributed_layout.DistributedLayout):
  """Distributed program definition for AIL."""

  def __init__(self,
               environment_factory: Callable[[bool], dm_env.Environment],
               rl_agent: builders.GenericActorLearnerBuilder,
               config: ail_config.AILConfig,
               network_factory: NetworkFactory,
               seed: int,
               batch_size: int,
               make_demonstrations: Callable[[int], Iterator[types.Transition]],
               policy_network: Any,
               evaluator_policy_network: Any,
               num_actors: int,
               max_number_of_steps: Optional[int] = None,
               log_to_bigtable: bool = False,
               log_every: float = 10.0,
               prefetch_size: int = 4,
               discriminator_loss: Optional[losses.Loss] = None):
    assert discriminator_loss is not None
    logger_fn = functools.partial(
        loggers.make_default_logger,
        'learner',
        log_to_bigtable,
        time_delta=log_every,
        asynchronous=True,
        serialize_fn=utils.fetch_devicearray,
        steps_key='learner_steps')
    ail_builder = builder.AILBuilder(
        rl_agent=rl_agent,
        config=config,
        discriminator_loss=discriminator_loss,
        make_demonstrations=make_demonstrations,
        logger_fn=logger_fn)
    super().__init__(
        seed=seed,
        environment_factory=lambda: environment_factory(False),
        network_factory=network_factory,
        builder=ail_builder,
        policy_network=policy_network,
        evaluator_factories=[
            distributed_layout.default_evaluator(
                environment_factory=lambda: environment_factory(True),
                network_factory=network_factory,
                builder=ail_builder,
                policy_factory=evaluator_policy_network,
                log_to_bigtable=log_to_bigtable)
        ],
        num_actors=num_actors,
        max_number_of_steps=max_number_of_steps,
        prefetch_size=prefetch_size,
        log_to_bigtable=log_to_bigtable,
        actor_logger_fn=distributed_layout.get_default_logger_fn(
            log_to_bigtable, log_every),
    )


class AIL(local_layout.LocalLayout):
  """Local agent for AIL."""

  def __init__(self,
               spec: specs.EnvironmentSpec,
               rl_agent: builders.GenericActorLearnerBuilder,
               network: ail_networks.AILNetworks,
               config: ail_config.AILConfig,
               seed: int,
               batch_size: int,
               make_demonstrations: Callable[[int], Iterator[types.Transition]],
               policy_network: Any,
               samples_per_insert: float = 256,
               discriminator_loss: Optional[losses.Loss] = None,
               counter: Optional[counting.Counter] = None):
    self.builder = builder.AILBuilder(
        rl_agent=rl_agent,
        config=config,
        discriminator_loss=discriminator_loss,
        make_demonstrations=make_demonstrations)
    super().__init__(
        seed=seed,
        environment_spec=spec,
        builder=self.builder,
        networks=network,
        policy_network=policy_network,
        batch_size=batch_size,
        samples_per_insert=samples_per_insert,
        min_replay_size=config.min_replay_size,
        num_sgd_steps_per_step=config.num_sgd_steps_per_step,
        counter=counter,
    )
