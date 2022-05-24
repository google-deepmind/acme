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

"""Defines distributed and local RND agents, using JAX."""

import functools
from typing import Callable, Generic, Optional, Sequence

from acme import specs
from acme.agents.jax import builders
from acme.agents.jax.rnd import builder
from acme.agents.jax.rnd import config as rnd_config
from acme.agents.jax.rnd import networks
from acme.jax import types as jax_types
from acme.jax import utils
from acme.jax.layouts import distributed_layout
from acme.jax.layouts import local_layout
from acme.jax.types import PolicyNetwork
from acme.utils import counting
from acme.utils import loggers
import reverb

NetworkFactory = Callable[[specs.EnvironmentSpec], networks.RNDNetworks]


class DistributedRND(Generic[networks.DirectRLNetworks, PolicyNetwork],
                     distributed_layout.DistributedLayout):
  """Distributed program definition for RND."""

  def __init__(
      self,
      environment_factory: jax_types.EnvironmentFactory,
      rl_agent: builders.ActorLearnerBuilder[networks.DirectRLNetworks,
                                             PolicyNetwork,
                                             reverb.ReplaySample],
      network_factory: NetworkFactory,
      config: rnd_config.RNDConfig,
      policy_network: PolicyNetwork,
      seed: int,
      num_actors: int,
      max_number_of_steps: Optional[int] = None,
      save_logs: bool = False,
      log_every: float = 10.0,
      prefetch_size: int = 4,
      evaluator_policy_network: Optional[PolicyNetwork] = None,
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
    rnd_builder = builder.RNDBuilder(rl_agent, config)
    if (evaluator_policy_network is None) == (evaluator_factories is None):
      raise ValueError('Either evaluator_policy_network or '
                       'evaluator_factories must be specified, but not both.')
    if evaluator_factories is None:
      evaluator_factories = [
          distributed_layout.default_evaluator_factory(
              environment_factory=environment_factory,
              network_factory=network_factory,
              policy_factory=evaluator_policy_network,
              save_logs=save_logs)
      ]
    super().__init__(
        seed=seed,
        environment_factory=environment_factory,
        learner_logger_fn=logger_fn,
        network_factory=network_factory,
        builder=rnd_builder,
        policy_network=policy_network,
        evaluator_factories=evaluator_factories,
        num_actors=num_actors,
        max_number_of_steps=max_number_of_steps,
        prefetch_size=prefetch_size,
        save_logs=save_logs,
        actor_logger_fn=distributed_layout.get_default_logger_fn(
            save_logs, log_every),
    )


class RND(Generic[networks.DirectRLNetworks, PolicyNetwork],
          local_layout.LocalLayout):
  """Local agent for RND."""

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      rl_agent: builders.ActorLearnerBuilder[networks.DirectRLNetworks,
                                             PolicyNetwork,
                                             reverb.ReplaySample],
      network: networks.RNDNetworks,
      config: rnd_config.RNDConfig,
      policy_network: PolicyNetwork,
      seed: int,
      batch_size: int = 256,
      min_replay_size: int = 10000,
      samples_per_insert: float = 256,
      counter: Optional[counting.Counter] = None,
  ):
    self.builder = builder.RNDBuilder(rl_agent, config)
    super().__init__(
        seed=seed,
        environment_spec=spec,
        builder=self.builder,
        networks=network,
        policy_network=policy_network,
        batch_size=batch_size,
        num_sgd_steps_per_step=config.num_sgd_steps_per_step,
        counter=counter,
    )
