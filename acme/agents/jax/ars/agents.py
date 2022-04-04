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

"""Defines distributed and local ARS agents, using JAX."""

import functools
from typing import Callable, Optional, Sequence

from acme import specs
from acme.agents.jax.ars import builder
from acme.agents.jax.ars import config as ars_config
from acme.agents.jax.ars import learning
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.jax import utils
from acme.jax.layouts import distributed_layout
from acme.utils import loggers

NetworkFactory = Callable[[specs.EnvironmentSpec],
                          networks_lib.FeedForwardNetwork]


class DistributedARS(distributed_layout.DistributedLayout):
  """Distributed program definition for ARS.
  """

  def __init__(
      self,
      environment_factory: jax_types.EnvironmentFactory,
      network_factory: NetworkFactory,
      config: ars_config.ARSConfig,
      seed: int,
      num_actors: int,
      max_number_of_steps: Optional[int] = None,
      log_to_bigtable: bool = False,
      log_every: float = 10.0,
      evaluator_factories: Optional[Sequence[
          distributed_layout.EvaluatorFactory]] = None,
  ):
    logger_fn = functools.partial(
        loggers.make_default_logger,
        'learner',
        log_to_bigtable,
        time_delta=log_every,
        asynchronous=True,
        serialize_fn=utils.fetch_devicearray,
        steps_key='learner_steps')
    dummy_seed = 1
    ars_builder = builder.ARSBuilder(
        config,
        spec=specs.make_environment_spec(environment_factory(dummy_seed)),
        logger_fn=logger_fn)
    if evaluator_factories is None:
      evaluator_factories = [
          distributed_layout.default_evaluator_factory(
              environment_factory=environment_factory,
              network_factory=network_factory,
              policy_factory=lambda n: (learning.EVAL_PARAMS_NAME, n),
              log_to_bigtable=log_to_bigtable)
      ]
    super().__init__(
        seed=seed,
        environment_factory=environment_factory,
        network_factory=network_factory,
        builder=ars_builder,
        policy_network=lambda n: (learning.BEHAVIOR_PARAMS_NAME, n),
        evaluator_factories=evaluator_factories,
        num_actors=num_actors,
        max_number_of_steps=max_number_of_steps,
        prefetch_size=0,
        log_to_bigtable=log_to_bigtable,
        actor_logger_fn=distributed_layout.get_default_logger_fn(
            log_to_bigtable, log_every),
    )
