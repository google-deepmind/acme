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
from typing import Callable, Optional, Sequence

from acme import specs
from acme.agents.jax.td3 import builder
from acme.agents.jax.td3 import config as td3_config
from acme.agents.jax.td3 import networks
from acme.jax import experiments
from acme.jax import types as jax_types
from acme.jax import utils
from acme.jax.layouts import distributed_layout
from acme.jax.layouts import local_layout
from acme.utils import counting
from acme.utils import loggers
import launchpad as lp

NetworkFactory = Callable[[specs.EnvironmentSpec], networks.TD3Networks]


# TODO(stanczyk): Remove DistributedTD3 once not used.
class DistributedTD3(distributed_layout.DistributedLayout):
  """Distributed program definition for TD3.

    DEPRECATED: Use distributed_td3 function instead.
  """

  def __init__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs

  def build(self, name='agent', program: Optional[lp.Program] = None):
    """Build the distributed agent topology."""
    return make_distributed_td3(
        *self.args, name=name, program=program, **self.kwargs)


def make_distributed_td3(environment_factory: jax_types.EnvironmentFactory,
                         environment_spec: specs.EnvironmentSpec,
                         network_factory: NetworkFactory,
                         config: td3_config.TD3Config,
                         seed: int,
                         num_actors: int,
                         max_number_of_steps: Optional[int] = None,
                         log_to_bigtable: bool = False,
                         log_every: float = 10.0,
                         evaluator_factories: Optional[Sequence[
                             distributed_layout.EvaluatorFactory]] = None,
                         name: str = 'agent',
                         program: Optional[lp.Program] = None):
  """Builds distributed TD3 program."""
  logger_fn = functools.partial(
      loggers.make_default_logger,
      'learner',
      log_to_bigtable,
      time_delta=log_every,
      asynchronous=True,
      serialize_fn=utils.fetch_devicearray,
      steps_key='learner_steps')
  td3_builder = builder.TD3Builder(config, logger_fn=logger_fn)

  action_specs = environment_spec.actions
  policy_network_fn = functools.partial(
      networks.get_default_behavior_policy,
      action_specs=action_specs,
      sigma=config.sigma)

  if evaluator_factories is None:
    eval_network_fn = functools.partial(
        networks.get_default_behavior_policy,
        action_specs=action_specs,
        sigma=0.)
    evaluator_factories = [
        distributed_layout.default_evaluator_factory(
            environment_factory=environment_factory,
            network_factory=network_factory,
            policy_factory=eval_network_fn,
            log_to_bigtable=log_to_bigtable)
    ]
  experiment = experiments.Config(
      builder=td3_builder,
      environment_factory=environment_factory,
      network_factory=network_factory,
      policy_network_factory=policy_network_fn,
      evaluator_factories=evaluator_factories,
      seed=seed,
      max_number_of_steps=max_number_of_steps,
      save_logs=log_to_bigtable)
  return experiments.make_distributed_experiment(
      experiment=experiment,
      num_actors=num_actors,
      prefetch_size=config.prefetch_size,
      actor_logger_fn=distributed_layout.get_default_logger_fn(
          log_to_bigtable, log_every),
      name=name,
      program=program)


class TD3(local_layout.LocalLayout):
  """Local agent for TD3."""

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      network: networks.TD3Networks,
      config: td3_config.TD3Config,
      seed: int,
      counter: Optional[counting.Counter] = None,
  ):
    behavior_policy = networks.get_default_behavior_policy(
        networks=network, action_specs=spec.actions, sigma=config.sigma)

    self.builder = builder.TD3Builder(config)
    super().__init__(
        seed=seed,
        environment_spec=spec,
        builder=self.builder,
        networks=network,
        policy_network=behavior_policy,
        batch_size=config.batch_size,
        prefetch_size=config.prefetch_size,
        num_sgd_steps_per_step=config.num_sgd_steps_per_step,
        counter=counter,
    )
