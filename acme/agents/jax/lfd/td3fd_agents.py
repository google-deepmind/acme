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

"""Agents combining Lfd with some typical offpolicy agents."""

import dataclasses
import functools
from typing import Callable, Iterator, Optional, Sequence

from acme import specs
from acme.agents.jax import td3
from acme.agents.jax.lfd import builder
from acme.agents.jax.lfd import config
from acme.jax import types as jax_types
from acme.jax import utils
from acme.jax.layouts import distributed_layout
from acme.jax.layouts import local_layout
from acme.utils import counting
from acme.utils import loggers

NetworkFactory = Callable[[specs.EnvironmentSpec], td3.TD3Networks]


@dataclasses.dataclass
class TD3fDConfig:
  """Configuration options specific to TD3 with demonstrations.

  Attributes:
    lfd_config: LfD config.
    td3_config: TD3 config.
  """
  lfd_config: config.LfdConfig
  td3_config: td3.TD3Config


class TD3fD(local_layout.LocalLayout):
  """TD3 agent learning from demonstrations."""

  def __init__(self,
               spec: specs.EnvironmentSpec,
               td3_network: td3.TD3Networks,
               td3_fd_config: TD3fDConfig,
               lfd_iterator_fn: Callable[[], Iterator[builder.LfdStep]],
               seed: int,
               counter: Optional[counting.Counter] = None):
    """New instance of a TD3fD agent."""
    td3_config = td3_fd_config.td3_config
    lfd_config = td3_fd_config.lfd_config
    td3_builder = td3.TD3Builder(td3_config)
    lfd_builder = builder.LfdBuilder(td3_builder, lfd_iterator_fn, lfd_config)

    behavior_policy = td3.get_default_behavior_policy(
        networks=td3_network, action_specs=spec.actions, sigma=td3_config.sigma)

    self.builder = lfd_builder
    super().__init__(
        seed=seed,
        environment_spec=spec,
        builder=lfd_builder,
        networks=td3_network,
        policy_network=behavior_policy,
        batch_size=td3_config.batch_size,
        prefetch_size=td3_config.prefetch_size,
        num_sgd_steps_per_step=td3_config.num_sgd_steps_per_step,
        counter=counter,
    )


class DistributedTD3fD(distributed_layout.DistributedLayout):
  """Distributed program definition for TD3 from demonstrations."""

  def __init__(
      self,
      environment_factory: jax_types.EnvironmentFactory,
      network_factory: NetworkFactory,
      td3_fd_config: TD3fDConfig,
      lfd_iterator_fn: Callable[[], Iterator[builder.LfdStep]],
      seed: int,
      num_actors: int,
      environment_spec: specs.EnvironmentSpec,
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

    td3_config = td3_fd_config.td3_config
    lfd_config = td3_fd_config.lfd_config
    td3_builder = td3.TD3Builder(td3_config)
    lfd_builder = builder.LfdBuilder(td3_builder, lfd_iterator_fn, lfd_config)

    action_specs = environment_spec.actions
    policy_network_fn = functools.partial(
        td3.get_default_behavior_policy,
        action_specs=action_specs,
        sigma=td3_config.sigma)

    if evaluator_factories is None:
      eval_network_fn = functools.partial(
          td3.get_default_behavior_policy, action_specs=action_specs, sigma=0.)
      evaluator_factories = [
          distributed_layout.default_evaluator_factory(
              environment_factory=environment_factory,
              network_factory=network_factory,
              policy_factory=eval_network_fn,
              save_logs=save_logs)
      ]
    super().__init__(
        seed=seed,
        environment_factory=environment_factory,
        learner_logger_fn=logger_fn,
        network_factory=network_factory,
        builder=lfd_builder,
        policy_network=policy_network_fn,
        evaluator_factories=evaluator_factories,
        num_actors=num_actors,
        max_number_of_steps=max_number_of_steps,
        prefetch_size=td3_config.prefetch_size,
        save_logs=save_logs,
        actor_logger_fn=distributed_layout.get_default_logger_fn(
            save_logs, log_every),
    )
