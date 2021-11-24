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

"""Defines distributed and local TD3 agents, using JAX."""

import functools
from typing import Callable, Optional

from acme import specs
from acme.agents.jax.td3 import builder
from acme.agents.jax.td3 import config as td3_config
from acme.agents.jax.td3 import networks
from acme.jax import utils
from acme.jax.layouts import distributed_layout
from acme.jax.layouts import local_layout
from acme.utils import counting
from acme.utils import loggers
import dm_env


NetworkFactory = Callable[[specs.EnvironmentSpec], networks.TD3Networks]


class DistributedTD3(distributed_layout.DistributedLayout):
  """Distributed program definition for TD3.
  """

  def __init__(
      self,
      environment_factory: Callable[[bool], dm_env.Environment],
      environment_spec: specs.EnvironmentSpec,
      network_factory: NetworkFactory,
      config: td3_config.TD3Config,
      seed: int,
      num_actors: int,
      max_number_of_steps: Optional[int] = None,
      log_to_bigtable: bool = False,
      log_every: float = 10.0,
  ):
    logger_fn = functools.partial(loggers.make_default_logger,
                                  'learner', log_to_bigtable,
                                  time_delta=log_every, asynchronous=True,
                                  serialize_fn=utils.fetch_devicearray,
                                  steps_key='learner_steps')
    td3_builder = builder.TD3Builder(config, logger_fn=logger_fn)

    action_specs = environment_spec.actions
    policy_network_fn = functools.partial(networks.get_default_behavior_policy,
                                          action_specs=action_specs,
                                          sigma=config.sigma)

    eval_network_fn = functools.partial(networks.get_default_behavior_policy,
                                        action_specs=action_specs,
                                        sigma=0.)
    super().__init__(
        seed=seed,
        environment_factory=lambda: environment_factory(False),
        network_factory=network_factory,
        builder=td3_builder,
        policy_network=policy_network_fn,
        evaluator_factories=[
            distributed_layout.default_evaluator(
                environment_factory=lambda: environment_factory(True),
                network_factory=network_factory,
                builder=td3_builder,
                policy_factory=eval_network_fn,
                log_to_bigtable=log_to_bigtable)
        ],
        num_actors=num_actors,
        max_number_of_steps=max_number_of_steps,
        prefetch_size=config.prefetch_size,
        log_to_bigtable=log_to_bigtable,
        actor_logger_fn=distributed_layout.get_default_logger_fn(
            log_to_bigtable, log_every),
    )


class TD3(local_layout.LocalLayout):
  """Local agent for TD3.
  """

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      network: networks.TD3Networks,
      config: td3_config.TD3Config,
      seed: int,
      counter: Optional[counting.Counter] = None,
  ):
    min_replay_size = config.min_replay_size
    # Local layout (actually agent.Agent) makes sure that we populate the
    # buffer with min_replay_size initial transitions and that there's no need
    # for tolerance_rate. In order to avoid deadlocks, we disable rate limiting
    # that is configured in TD3Builder.make_replay_tables. This is achieved by
    # the following two lines.
    config.samples_per_insert_tolerance_rate = float('inf')
    config.min_replay_size = 1

    behavior_policy = networks.get_default_behavior_policy(
        networks=network,
        action_specs=spec.actions,
        sigma=config.sigma)

    self.builder = builder.TD3Builder(config)
    super().__init__(
        seed=seed,
        environment_spec=spec,
        builder=self.builder,
        networks=network,
        policy_network=behavior_policy,
        batch_size=config.batch_size,
        prefetch_size=config.prefetch_size,
        samples_per_insert=config.samples_per_insert,
        min_replay_size=min_replay_size,
        num_sgd_steps_per_step=config.num_sgd_steps_per_step,
        counter=counter,
    )
