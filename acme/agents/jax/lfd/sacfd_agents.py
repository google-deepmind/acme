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
from acme.agents.jax import sac
from acme.agents.jax.lfd import builder
from acme.agents.jax.lfd import config
from acme.jax import types as jax_types
from acme.jax import utils
from acme.jax.layouts import distributed_layout
from acme.jax.layouts import local_layout
from acme.utils import counting
from acme.utils import loggers


NetworkFactory = Callable[[specs.EnvironmentSpec], sac.SACNetworks]


@dataclasses.dataclass
class SACFDConfig:
  """Configuration options specific to SAC with demonstrations.

  Attributes:
    lfd_config: LfD config.
    sac_config: SAC config.
  """
  lfd_config: config.LfdConfig
  sac_config: sac.SACConfig


class SACFD(local_layout.LocalLayout):
  """SAC agent learning from demonstrations."""

  def __init__(self,
               spec: specs.EnvironmentSpec,
               sac_network: sac.SACNetworks,
               sac_fd_config: SACFDConfig,
               lfd_iterator_fn: Callable[[], Iterator[builder.LfdStep]],
               seed: int,
               counter: Optional[counting.Counter] = None):
    """New instance of a SAC-Fd agent."""
    sac_config = sac_fd_config.sac_config
    lfd_config = sac_fd_config.lfd_config
    sac_builder = sac.SACBuilder(sac_config)
    lfd_builder = builder.LfdBuilder(sac_builder, lfd_iterator_fn, lfd_config)

    min_replay_size = sac_config.min_replay_size
    # Local layout (actually agent.Agent) makes sure that we populate the
    # buffer with min_replay_size initial transitions and that there's no need
    # for tolerance_rate. In order for deadlocks not to happen we need to
    # disable rate limiting that heppens inside the SACBuilder. This is achieved
    # by the following two lines.
    sac_config.samples_per_insert_tolerance_rate = float('inf')
    sac_config.min_replay_size = 1

    self.builder = lfd_builder
    super().__init__(
        builder=lfd_builder,
        seed=seed,
        environment_spec=spec,
        networks=sac_network,
        policy_network=sac.apply_policy_and_sample(sac_network),
        batch_size=sac_config.batch_size,
        prefetch_size=sac_config.prefetch_size,
        samples_per_insert=sac_config.samples_per_insert,
        min_replay_size=min_replay_size,
        num_sgd_steps_per_step=sac_config.num_sgd_steps_per_step,
        counter=counter,
        )


class DistributedSACFD(distributed_layout.DistributedLayout):
  """Distributed program definition for SAC from demonstrations."""

  def __init__(
      self,
      environment_factory: jax_types.EnvironmentFactory,
      network_factory: NetworkFactory,
      sac_fd_config: SACFDConfig,
      lfd_iterator_fn: Callable[[], Iterator[builder.LfdStep]],
      seed: int,
      num_actors: int,
      environment_spec: Optional[specs.EnvironmentSpec] = None,
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

    sac_config = sac_fd_config.sac_config
    lfd_config = sac_fd_config.lfd_config
    sac_builder = sac.SACBuilder(sac_config, logger_fn=logger_fn)
    lfd_builder = builder.LfdBuilder(sac_builder, lfd_iterator_fn, lfd_config)

    if evaluator_factories is None:
      eval_policy_factory = (
          lambda n: sac.apply_policy_and_sample(n, True))
      evaluator_factories = [
          distributed_layout.default_evaluator_factory(
              environment_factory=environment_factory,
              network_factory=network_factory,
              policy_factory=eval_policy_factory,
              log_to_bigtable=log_to_bigtable)
      ]

    super().__init__(
        seed=seed,
        environment_factory=environment_factory,
        network_factory=network_factory,
        environment_spec=environment_spec,
        builder=lfd_builder,
        policy_network=sac.apply_policy_and_sample,
        evaluator_factories=evaluator_factories,
        num_actors=num_actors,
        max_number_of_steps=max_number_of_steps,
        prefetch_size=sac_config.prefetch_size,
        log_to_bigtable=log_to_bigtable,
        actor_logger_fn=distributed_layout.get_default_logger_fn(
            log_to_bigtable, log_every))
