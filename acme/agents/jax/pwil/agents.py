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

"""Defines distributed and local PWIL agents, using JAX."""

from typing import Callable, Generic, Optional, Sequence

from acme import specs
from acme.agents.jax import builders
from acme.agents.jax.pwil import builder
from acme.agents.jax.pwil import config as pwil_config
from acme.jax import types as jax_types
from acme.jax.imitation_learning_types import DirectPolicyNetwork, DirectRLNetworks, DirectRLTrainingState  # pylint: disable=g-multiple-import
from acme.jax.layouts import distributed_layout
from acme.jax.layouts import local_layout
from acme.utils import counting
import reverb


class DistributedPWIL(distributed_layout.DistributedLayout,
                      Generic[DirectRLNetworks, DirectPolicyNetwork,
                              DirectRLTrainingState]):
  """Distributed program definition for PWIL."""

  def __init__(
      self,
      environment_factory: jax_types.EnvironmentFactory,
      rl_agent: builders.ActorLearnerBuilder[DirectRLNetworks,
                                             DirectPolicyNetwork,
                                             reverb.ReplaySample],
      config: pwil_config.PWILConfig,
      network_factory: Callable[[specs.EnvironmentSpec], DirectRLNetworks],
      seed: int,
      demonstrations_fn: Callable[[], pwil_config.PWILDemonstrations],
      policy_network: Callable[[DirectRLNetworks], DirectPolicyNetwork],
      num_actors: int,
      max_number_of_steps: Optional[int] = None,
      save_logs: bool = False,
      log_every: float = 10.0,
      prefetch_size: int = 4,
      evaluator_policy_network: Optional[Callable[[DirectRLNetworks],
                                                  DirectPolicyNetwork]] = None,
      evaluator_factories: Optional[Sequence[
          distributed_layout.EvaluatorFactory]] = None,
  ):
    pwil_builder = builder.PWILBuilder(
        rl_agent=rl_agent, config=config, demonstrations_fn=demonstrations_fn)
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
        builder=pwil_builder,
        policy_network=policy_network,
        evaluator_factories=evaluator_factories,
        num_actors=num_actors,
        max_number_of_steps=max_number_of_steps,
        prefetch_size=prefetch_size,
        save_logs=save_logs,
        actor_logger_fn=distributed_layout.get_default_logger_fn(
            save_logs, log_every),
    )


class PWIL(local_layout.LocalLayout,
           Generic[DirectRLNetworks, DirectPolicyNetwork,
                   DirectRLTrainingState]):
  """Local agent for PWIL."""

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      rl_agent: builders.ActorLearnerBuilder[DirectRLNetworks,
                                             DirectPolicyNetwork,
                                             reverb.ReplaySample],
      config: pwil_config.PWILConfig,
      networks: DirectRLNetworks,
      seed: int,
      batch_size: int,
      demonstrations_fn: Callable[[], pwil_config.PWILDemonstrations],
      policy_network: DirectPolicyNetwork,
      counter: Optional[counting.Counter] = None,
  ):
    self.builder = builder.PWILBuilder(
        rl_agent=rl_agent, config=config, demonstrations_fn=demonstrations_fn)
    super().__init__(
        seed=seed,
        environment_spec=spec,
        builder=self.builder,
        networks=networks,
        policy_network=policy_network,
        batch_size=batch_size,
        num_sgd_steps_per_step=config.num_sgd_steps_per_step,
        counter=counter,
    )
