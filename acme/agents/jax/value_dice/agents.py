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

"""Defines distributed and local ValueDice agents, using JAX."""

import functools
from typing import Callable, Iterator, Optional

from acme import specs
from acme import types
from acme.agents.jax.value_dice import builder
from acme.agents.jax.value_dice import config as value_dice_config
from acme.agents.jax.value_dice import networks
from acme.jax import utils
from acme.jax.layouts import distributed_layout
from acme.jax.layouts import local_layout
from acme.utils import counting
from acme.utils import loggers
import dm_env


NetworkFactory = Callable[[specs.EnvironmentSpec], networks.ValueDiceNetworks]


class DistributedValueDice(distributed_layout.DistributedLayout):
  """Distributed program definition for ValueDice.
  """

  def __init__(
      self,
      environment_factory: Callable[[bool], dm_env.Environment],
      network_factory: NetworkFactory,
      config: value_dice_config.ValueDiceConfig,
      make_demonstrations: Callable[[int], Iterator[types.Transition]],
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
    spec = specs.make_environment_spec(environment_factory(False))
    value_dice_builder = builder.ValueDiceBuilder(
        config=config, logger_fn=logger_fn,
        make_demonstrations=make_demonstrations)
    eval_policy_factory = (
        lambda n: networks.apply_policy_and_sample(n, True))
    super().__init__(
        seed=seed,
        environment_spec=spec,
        environment_factory=lambda: environment_factory(False),
        network_factory=network_factory,
        builder=value_dice_builder,
        policy_network=networks.apply_policy_and_sample,
        evaluator_factories=[
            distributed_layout.default_evaluator(
                environment_factory=lambda: environment_factory(True),
                network_factory=network_factory,
                builder=value_dice_builder,
                policy_factory=eval_policy_factory,
                log_to_bigtable=log_to_bigtable)
        ],
        num_actors=num_actors,
        max_number_of_steps=max_number_of_steps,
        prefetch_size=config.prefetch_size,
        log_to_bigtable=log_to_bigtable,
        actor_logger_fn=distributed_layout.get_default_logger_fn(
            log_to_bigtable, log_every),
    )


class ValueDice(local_layout.LocalLayout):
  """Local agent for ValueDice.
  """

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      network: networks.ValueDiceNetworks,
      config: value_dice_config.ValueDiceConfig,
      make_demonstrations: Callable[[int], Iterator[types.Transition]],
      seed: int,
      counter: Optional[counting.Counter] = None,
  ):
    min_replay_size = config.min_replay_size
    # Local layout (actually agent.Agent) makes sure that we populate the
    # buffer with min_replay_size initial transitions and that there's no need
    # for tolerance_rate. In order for deadlocks not to happen we need to
    # disable rate limiting that heppens inside the ValueDiceBuilder.
    # This is achieved by the following two lines.
    config.samples_per_insert_tolerance_rate = float('inf')
    config.min_replay_size = 1
    self.builder = builder.ValueDiceBuilder(
        config=config, make_demonstrations=make_demonstrations)
    super().__init__(
        seed=seed,
        environment_spec=spec,
        builder=self.builder,
        networks=network,
        policy_network=networks.apply_policy_and_sample(network),
        batch_size=config.batch_size,
        samples_per_insert=config.samples_per_insert,
        min_replay_size=min_replay_size,
        num_sgd_steps_per_step=config.num_sgd_steps_per_step,
        counter=counter,
    )
