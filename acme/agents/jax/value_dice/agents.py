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
from typing import Callable, Iterator, Optional, Sequence

from acme import specs
from acme import types
from acme.agents.jax.value_dice import builder
from acme.agents.jax.value_dice import config as value_dice_config
from acme.agents.jax.value_dice import networks
from acme.jax import types as jax_types
from acme.jax import utils
from acme.jax.layouts import distributed_layout
from acme.jax.layouts import local_layout
from acme.utils import counting
from acme.utils import loggers

NetworkFactory = Callable[[specs.EnvironmentSpec], networks.ValueDiceNetworks]


class DistributedValueDice(distributed_layout.DistributedLayout):
  """Distributed program definition for ValueDice."""

  def __init__(
      self,
      environment_factory: jax_types.EnvironmentFactory,
      network_factory: NetworkFactory,
      config: value_dice_config.ValueDiceConfig,
      make_demonstrations: Callable[[int], Iterator[types.Transition]],
      seed: int,
      num_actors: int,
      max_number_of_steps: Optional[int] = None,
      save_logs: bool = False,
      log_every: float = 10.0,
      evaluator_factories: Optional[Sequence[
          distributed_layout.EvaluatorFactory]] = None,
  ):
    logger_fn = functools.partial(
        loggers.make_default_logger,
        'learner',
        save_logs,
        time_delta=log_every,
        asynchronous=True,
        serialize_fn=utils.fetch_devicearray,
        steps_key='learner_steps')
    dummy_seed = 1
    spec = specs.make_environment_spec(environment_factory(dummy_seed))
    value_dice_builder = builder.ValueDiceBuilder(
        config=config,
        make_demonstrations=make_demonstrations)
    if evaluator_factories is None:
      eval_policy_factory = (
          lambda n: networks.apply_policy_and_sample(n, True))
      evaluator_factories = [
          distributed_layout.default_evaluator_factory(
              environment_factory=environment_factory,
              network_factory=network_factory,
              policy_factory=eval_policy_factory,
              save_logs=save_logs)
      ]
    super().__init__(
        seed=seed,
        environment_spec=spec,
        environment_factory=environment_factory,
        learner_logger_fn=logger_fn,
        network_factory=network_factory,
        builder=value_dice_builder,
        policy_network=networks.apply_policy_and_sample,
        evaluator_factories=evaluator_factories,
        num_actors=num_actors,
        max_number_of_steps=max_number_of_steps,
        prefetch_size=config.prefetch_size,
        save_logs=save_logs,
        actor_logger_fn=distributed_layout.get_default_logger_fn(
            save_logs, log_every),
    )


class ValueDice(local_layout.LocalLayout):
  """Local agent for ValueDice."""

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      network: networks.ValueDiceNetworks,
      config: value_dice_config.ValueDiceConfig,
      make_demonstrations: Callable[[int], Iterator[types.Transition]],
      seed: int,
      counter: Optional[counting.Counter] = None,
  ):
    self.builder = builder.ValueDiceBuilder(
        config=config, make_demonstrations=make_demonstrations)
    super().__init__(
        seed=seed,
        environment_spec=spec,
        builder=self.builder,
        networks=network,
        policy_network=networks.apply_policy_and_sample(network),
        batch_size=config.batch_size,
        num_sgd_steps_per_step=config.num_sgd_steps_per_step,
        counter=counter,
    )
