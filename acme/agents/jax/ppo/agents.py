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

"""Defines distributed and local PPO agents, using JAX."""

import functools
from typing import Callable, Optional, Sequence

from acme import specs
from acme.agents.jax import normalization
from acme.agents.jax.ppo import builder
from acme.agents.jax.ppo import config as ppo_config
from acme.agents.jax.ppo import networks as ppo_networks
from acme.jax import experiments
from acme.jax import types as jax_types
from acme.jax import utils
from acme.jax.layouts import distributed_layout
from acme.jax.layouts import local_layout
from acme.utils import counting
from acme.utils import loggers

import launchpad as lp
NetworkFactory = Callable[[specs.EnvironmentSpec], ppo_networks.PPONetworks]


def make_distributed_ppo(environment_factory: jax_types.EnvironmentFactory,
                         network_factory: NetworkFactory,
                         config: ppo_config.PPOConfig,
                         seed: int,
                         num_actors: int,
                         normalize_input: bool = False,
                         logger_fn: Optional[Callable[[],
                                                      loggers.Logger]] = None,
                         save_reverb_logs: bool = False,
                         log_every: float = 10.0,
                         max_number_of_steps: Optional[int] = None,
                         evaluator_factories: Optional[Sequence[
                             experiments.EvaluatorFactory]] = None,
                         make_snapshot_models: Optional[
                             distributed_layout.SnapshotModelFactory] = None,
                         name='agent',
                         program: Optional[lp.Program] = None):
  """Builds distributed PPO program."""
  logger_fn = logger_fn or functools.partial(
      loggers.make_default_logger,
      'learner',
      save_reverb_logs,
      time_delta=log_every,
      asynchronous=True,
      serialize_fn=utils.fetch_devicearray,
      steps_key='learner_steps')
  ppo_builder = builder.PPOBuilder(config)
  if normalize_input:
    dummy_seed = 1
    environment_spec = specs.make_environment_spec(
        environment_factory(dummy_seed))
    # Two batch dimensions: [num_sequences, num_steps, ...]
    batch_dims = (0, 1)
    ppo_builder = normalization.NormalizationBuilder(
        ppo_builder,
        environment_spec,
        is_sequence_based=True,
        batch_dims=batch_dims)
  experiment = experiments.Config(
      builder=ppo_builder,
      environment_factory=environment_factory,
      network_factory=network_factory,
      policy_network_factory=ppo_networks.make_inference_fn,
      evaluator_factories=evaluator_factories,
      eval_policy_network_factory=(
          lambda network: ppo_networks.make_inference_fn(network, True)),
      seed=seed,
      max_number_of_steps=max_number_of_steps,
      logger_factory=distributed_layout.logger_factory(logger_fn, None,
                                                       save_reverb_logs))

  return experiments.make_distributed_experiment(
      experiment=experiment,
      num_actors=num_actors,
      prefetch_size=config.prefetch_size,
      make_snapshot_models=make_snapshot_models,
      name=name,
      program=program)


class PPO(local_layout.LocalLayout):
  """Local agent for PPO."""

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      networks: ppo_networks.PPONetworks,
      config: ppo_config.PPOConfig,
      seed: int,
      workdir: Optional[str] = '~/acme',
      normalize_input: bool = False,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
  ):
    ppo_builder = builder.PPOBuilder(config)
    if normalize_input:
      # Two batch dimensions: [num_sequences, num_steps, ...]
      batch_dims = (0, 1)
      ppo_builder = normalization.NormalizationBuilder(
          ppo_builder, spec, is_sequence_based=True, batch_dims=batch_dims)
    self.builder = ppo_builder
    super().__init__(
        seed=seed,
        environment_spec=spec,
        builder=ppo_builder,
        networks=networks,
        learner_logger=logger,
        policy_network=ppo_networks.make_inference_fn(networks),
        batch_size=config.batch_size,
        # TODO(sinopalnikov): move it to the experiment config
        workdir=workdir,
        counter=counter,
    )
