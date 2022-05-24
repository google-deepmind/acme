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

"""Defines distributed and local SQIL agents, using JAX."""

from typing import Callable, Generic, Iterator, Optional, Sequence

from acme import specs
from acme import types
from acme.agents.jax import builders
from acme.agents.jax.sqil import builder
from acme.jax import types as jax_types
from acme.jax.layouts import distributed_layout
from acme.jax.layouts import local_layout
from acme.jax.types import Networks, PolicyNetwork  # pylint: disable=g-multiple-import
from acme.utils import counting
import reverb

NetworkFactory = Callable[[specs.EnvironmentSpec], Networks]


class DistributedSQIL(Generic[Networks, PolicyNetwork],
                      distributed_layout.DistributedLayout):
  """Distributed program definition for SQIL."""

  def __init__(
      self,
      environment_factory: jax_types.EnvironmentFactory,
      rl_agent: builders.ActorLearnerBuilder[Networks, PolicyNetwork,
                                             reverb.ReplaySample],
      network_factory: NetworkFactory,
      seed: int,
      batch_size: int,
      make_demonstrations: Callable[[int], Iterator[types.Transition]],
      policy_network: PolicyNetwork,
      num_actors: int,
      max_number_of_steps: Optional[int] = None,
      save_logs: bool = False,
      log_every: float = 10.0,
      prefetch_size: int = 4,
      evaluator_policy_network: Optional[PolicyNetwork] = None,
      evaluator_factories: Optional[Sequence[
          distributed_layout.EvaluatorFactory]] = None,
  ):
    sqil_builder = builder.SQILBuilder(rl_agent, batch_size,
                                       make_demonstrations)
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
        network_factory=network_factory,
        builder=sqil_builder,
        policy_network=policy_network,
        evaluator_factories=evaluator_factories,
        num_actors=num_actors,
        max_number_of_steps=max_number_of_steps,
        prefetch_size=prefetch_size,
        save_logs=save_logs,
        actor_logger_fn=distributed_layout.get_default_logger_fn(
            save_logs, log_every),
    )


class SQIL(Generic[Networks, PolicyNetwork], local_layout.LocalLayout):
  """Local agent for SQIL."""

  def __init__(self,
               spec: specs.EnvironmentSpec,
               rl_agent: builders.ActorLearnerBuilder[Networks, PolicyNetwork,
                                                      reverb.ReplaySample],
               network: Networks,
               seed: int,
               batch_size: int,
               make_demonstrations: Callable[[int], Iterator[types.Transition]],
               policy_network: PolicyNetwork,
               min_replay_size: int = 10000,
               samples_per_insert: float = 256,
               num_sgd_steps_per_step: int = 1,
               counter: Optional[counting.Counter] = None):
    self.builder = builder.SQILBuilder(rl_agent, batch_size,
                                       make_demonstrations)
    super().__init__(
        seed=seed,
        environment_spec=spec,
        builder=self.builder,
        networks=network,
        policy_network=policy_network,
        batch_size=batch_size,
        num_sgd_steps_per_step=num_sgd_steps_per_step,
        counter=counter,
    )
