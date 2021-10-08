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

"""Defines the D4PG agent class, using JAX."""

import functools
from typing import Callable, Optional

from acme import specs
from acme.agents.jax.d4pg import builder
from acme.jax import utils
from acme.jax.layouts import distributed_layout
from acme.jax.layouts import local_layout
from acme.utils import counting
from acme.utils import loggers
import dm_env


NetworkFactory = Callable[[specs.EnvironmentSpec], builder.D4PGNetworks]


class DistributedD4PG(distributed_layout.DistributedLayout):
  """Program definition for D4PG.

  We are in the process of migrating towards a more modular agent configuration.
  This is maintained now for compatibility.
  """

  def __init__(
      self,
      environment_factory: Callable[[bool], dm_env.Environment],
      network_factory: NetworkFactory,
      random_seed: int,
      num_actors: int,
      environment_spec: Optional[specs.EnvironmentSpec] = None,
      batch_size: int = 256,
      prefetch_size: int = 2,
      min_replay_size: int = 1000,
      max_replay_size: int = 1000000,
      samples_per_insert: float = 32.0,
      n_step: int = 5,
      sigma: float = 0.3,
      clipping: bool = True,
      discount: float = 0.99,
      target_update_period: int = 100,
      device_prefetch: bool = True,
      log_to_bigtable: bool = False,
      log_every: float = 10.0,
  ):
    config = builder.D4PGConfig(
        discount=discount,
        learning_rate=1e-4,
        batch_size=batch_size,
        prefetch_size=prefetch_size,
        target_update_period=target_update_period,
        min_replay_size=min_replay_size,
        max_replay_size=max_replay_size,
        samples_per_insert=samples_per_insert,
        n_step=n_step,
        sigma=sigma,
        clipping=clipping,
    )
    logger_fn = functools.partial(
        loggers.make_default_logger,
        'learner',
        log_to_bigtable,
        time_delta=log_every,
        asynchronous=True,
        serialize_fn=utils.fetch_devicearray,
        steps_key='learner_steps')
    d4pg_builder = builder.D4PGBuilder(config, logger_fn=logger_fn)

    super().__init__(
        seed=random_seed,
        environment_factory=lambda: environment_factory(False),
        network_factory=network_factory,
        builder=d4pg_builder,
        policy_network=lambda n: builder.get_default_behavior_policy(n, config),
        evaluator_factories=[
            distributed_layout.default_evaluator(
                environment_factory=lambda: environment_factory(True),
                network_factory=network_factory,
                builder=d4pg_builder,
                policy_factory=builder.get_default_eval_policy,
                log_to_bigtable=log_to_bigtable)
        ],
        num_actors=num_actors,
        environment_spec=environment_spec,
        device_prefetch=device_prefetch,
        log_to_bigtable=log_to_bigtable,
        actor_logger_fn=distributed_layout.get_default_logger_fn(
            log_to_bigtable, log_every),
        prefetch_size=config.prefetch_size,
    )


class D4PG(local_layout.LocalLayout):
  """Local agent for D4PG.
  """

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      network: builder.D4PGNetworks,
      config: builder.D4PGConfig,
      random_seed: int,
      counter: Optional[counting.Counter] = None,
  ):
    # In the case of a synchronous agent, we do not use Reverb's built-in rate
    # limitation to avoid deadlocks; so rather than using the Builder's min
    # replay size, we trivially set it to 1 after extracting the requested value
    # to be honored by the synchronous Layout instead.
    min_replay_size = config.min_replay_size
    config.min_replay_size = 1

    # Local layout (actually agent.Agent) makes sure that we populate the
    # buffer with min_replay_size initial transitions and that there's no need
    # for tolerance_rate. In order to avoid deadlocks we disable rate limiting.
    # This is achieved by setting the rate tolerance to be infinite.
    config.samples_per_insert_tolerance_rate = float('inf')

    self.builder = builder.D4PGBuilder(config)
    super().__init__(
        seed=random_seed,
        environment_spec=spec,
        builder=self.builder,
        networks=network,
        policy_network=builder.get_default_behavior_policy(network, config),
        batch_size=config.batch_size,
        samples_per_insert=config.samples_per_insert,
        min_replay_size=min_replay_size,
        num_sgd_steps_per_step=config.num_sgd_steps_per_step,
        counter=counter,
    )
