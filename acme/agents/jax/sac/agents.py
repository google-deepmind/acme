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

"""Defines distributed and local SAC agents, using JAX."""

import functools
from typing import Callable, Optional, Sequence

from acme import specs
from acme.agents.jax import normalization
from acme.agents.jax.sac import builder
from acme.agents.jax.sac import config as sac_config
from acme.agents.jax.sac import networks
from acme.jax import experiments
from acme.jax import types as jax_types
from acme.jax import utils
from acme.jax.layouts import distributed_layout
from acme.jax.layouts import local_layout
from acme.utils import counting
from acme.utils import loggers
import launchpad as lp

NetworkFactory = Callable[[specs.EnvironmentSpec], networks.SACNetworks]


# TODO(stanczyk): Remove DistributedTD3 once not used.
class DistributedSAC(distributed_layout.DistributedLayout):
  """Distributed program definition for SAC.

    DEPRECATED: Use distributed_sac function instead.
  """

  def __init__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs

  def build(self, name='agent', program: Optional[lp.Program] = None):
    """Build the distributed agent topology."""
    return make_distributed_sac(
        *self.args, name=name, program=program, **self.kwargs)


def make_distributed_sac(environment_factory: jax_types.EnvironmentFactory,
                         network_factory: NetworkFactory,
                         config: sac_config.SACConfig,
                         seed: int,
                         num_actors: int,
                         max_number_of_steps: Optional[int] = None,
                         save_logs: bool = False,
                         log_every: float = 10.0,
                         normalize_input: bool = True,
                         evaluator_factories: Optional[Sequence[
                             distributed_layout.EvaluatorFactory]] = None,
                         name: str = 'agent',
                         program: Optional[lp.Program] = None):
  """Builds distributed SAC program."""
  logger_fn = functools.partial(
      loggers.make_default_logger,
      'learner',
      save_logs,
      time_delta=log_every,
      asynchronous=True,
      serialize_fn=utils.fetch_devicearray,
      steps_key='learner_steps')
  sac_builder = builder.SACBuilder(config)
  if normalize_input:
    # One batch dimension: [batch_size, ...]
    batch_dims = (0,)
    sac_builder = normalization.NormalizationBuilder(
        sac_builder, is_sequence_based=False, batch_dims=batch_dims)
  if evaluator_factories is None:
    eval_policy_factory = (lambda n: networks.apply_policy_and_sample(n, True))
    evaluator_factories = [
        distributed_layout.default_evaluator_factory(
            environment_factory=environment_factory,
            network_factory=network_factory,
            policy_factory=eval_policy_factory,
            save_logs=save_logs)
    ]
  experiment = experiments.ExperimentConfig(
      builder=sac_builder,
      environment_factory=environment_factory,
      network_factory=network_factory,
      policy_network_factory=networks.apply_policy_and_sample,
      evaluator_factories=evaluator_factories,
      seed=seed,
      max_num_actor_steps=max_number_of_steps,
      logger_factory=distributed_layout.logger_factory(logger_fn, None,
                                                       save_logs, log_every))
  return experiments.make_distributed_experiment(
      experiment=experiment,
      num_actors=num_actors,
      checkpointing_config=distributed_layout.CheckpointingConfig(),
      make_snapshot_models=networks.default_models_to_snapshot,
      name=name,
      program=program)


class SAC(local_layout.LocalLayout):
  """Local agent for SAC."""

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      network: networks.SACNetworks,
      config: sac_config.SACConfig,
      seed: int,
      normalize_input: bool = True,
      counter: Optional[counting.Counter] = None,
  ):
    sac_builder = builder.SACBuilder(config)
    if normalize_input:
      # One batch dimension: [batch_size, ...]
      batch_dims = (0,)
      sac_builder = normalization.NormalizationBuilder(
          sac_builder, is_sequence_based=False, batch_dims=batch_dims)
    self.builder = sac_builder
    super().__init__(
        seed=seed,
        environment_spec=spec,
        builder=sac_builder,
        networks=network,
        policy_network=networks.apply_policy_and_sample(network),
        batch_size=config.batch_size,
        prefetch_size=config.prefetch_size,
        num_sgd_steps_per_step=config.num_sgd_steps_per_step,
        counter=counter,
    )
