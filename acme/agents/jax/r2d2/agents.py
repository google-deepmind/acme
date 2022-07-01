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

"""Defines distributed and local R2D2 agents, using JAX."""

import functools
from typing import Callable, Optional, Sequence

from acme import specs
from acme.agents.jax.r2d2 import actor as r2d2_actor
from acme.agents.jax.r2d2 import builder
from acme.agents.jax.r2d2 import config as r2d2_config
from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.jax import types as jax_types
from acme.jax import utils
from acme.jax.layouts import distributed_layout
from acme.jax.layouts import local_layout
from acme.utils import counting
from acme.utils import loggers
import haiku as hk
import rlax

NetworkFactory = Callable[[specs.EnvironmentSpec], r2d2_networks.R2D2Networks]


class DistributedR2D2FromConfig(distributed_layout.DistributedLayout):
  """Distributed R2D2 agents from config."""

  def __init__(
      self,
      environment_factory: jax_types.EnvironmentFactory,
      environment_spec: specs.EnvironmentSpec,
      network_factory: NetworkFactory,
      config: r2d2_config.R2D2Config,
      seed: int,
      num_actors: int,
      workdir: str = '~/acme',
      save_logs: bool = True,
      log_every: float = 10.0,
      evaluator_factories: Optional[Sequence[
          distributed_layout.EvaluatorFactory]] = None,
      max_number_of_steps: Optional[int] = None,
  ):
    logger_fn = functools.partial(
        loggers.make_default_logger,
        'learner',
        save_logs,
        time_delta=log_every,
        asynchronous=True,
        serialize_fn=utils.fetch_devicearray,
        steps_key='learner_steps')
    policy_network_factory = (
        lambda n: r2d2_actor.make_behavior_policy(n, config))
    if evaluator_factories is None:
      evaluator_policy_network_factory = (
          lambda n: r2d2_actor.make_behavior_policy(n, config, True))
      evaluator_factories = [
          distributed_layout.default_evaluator_factory(
              environment_factory=environment_factory,
              network_factory=network_factory,
              policy_factory=evaluator_policy_network_factory,
              save_logs=save_logs)
      ]
    super().__init__(
        seed=seed,
        environment_factory=environment_factory,
        network_factory=network_factory,
        builder=builder.R2D2Builder(config),
        policy_network=policy_network_factory,
        evaluator_factories=evaluator_factories,
        num_actors=num_actors,
        environment_spec=environment_spec,
        save_logs=save_logs,
        learner_logger_fn=logger_fn,
        actor_logger_fn=distributed_layout.get_default_logger_fn(
            save_logs, log_every),
        prefetch_size=config.prefetch_size,
        checkpointing_config=distributed_layout.CheckpointingConfig(
            directory=workdir, add_uid=(workdir == '~/acme')),
        max_number_of_steps=max_number_of_steps)


class DistributedR2D2(DistributedR2D2FromConfig):
  """Distributed R2D2 agent."""

  def __init__(
      self,
      environment_factory: jax_types.EnvironmentFactory,
      environment_spec: specs.EnvironmentSpec,
      forward: hk.Transformed,
      unroll: hk.Transformed,
      initial_state: hk.Transformed,
      num_actors: int,
      num_caches: int = 1,
      burn_in_length: int = 40,
      trace_length: int = 80,
      sequence_period: int = 40,
      batch_size: int = 64,
      prefetch_size: int = 2,
      target_update_period: int = 2500,
      samples_per_insert: float = 0.,
      min_replay_size: int = 1000,
      max_replay_size: int = 100_000,
      importance_sampling_exponent: float = 0.6,
      priority_exponent: float = 0.9,
      max_priority_weight: float = 0.9,
      bootstrap_n: int = 5,
      clip_rewards: bool = False,
      tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR,
      learning_rate: float = 1e-3,
      evaluator_epsilon: float = 0.,
      discount: float = 0.997,
      variable_update_period: int = 400,
      max_number_of_steps: Optional[int] = None,
      seed: int = 1,
  ):
    config = r2d2_config.R2D2Config(
        discount=discount,
        target_update_period=target_update_period,
        evaluation_epsilon=evaluator_epsilon,
        burn_in_length=burn_in_length,
        trace_length=trace_length,
        sequence_period=sequence_period,
        learning_rate=learning_rate,
        bootstrap_n=bootstrap_n,
        clip_rewards=clip_rewards,
        tx_pair=tx_pair,
        samples_per_insert=samples_per_insert,
        min_replay_size=min_replay_size,
        max_replay_size=max_replay_size,
        batch_size=batch_size,
        prefetch_size=prefetch_size,
        importance_sampling_exponent=importance_sampling_exponent,
        priority_exponent=priority_exponent,
        max_priority_weight=max_priority_weight,
    )
    network_factory = functools.partial(
        r2d2_networks.make_networks,
        forward_fn=forward,
        initial_state_fn=initial_state,
        unroll_fn=unroll,
        batch_size=batch_size)
    super().__init__(
        seed=seed,
        environment_factory=environment_factory,
        environment_spec=environment_spec,
        network_factory=network_factory,
        config=config,
        num_actors=num_actors,
        max_number_of_steps=max_number_of_steps,
    )


class R2D2(local_layout.LocalLayout):
  """Local agent for r2d2.

  This implements a single-process R2D2 agent. This is a simple Q-learning
  algorithm that generates data via a (epsilon-greedy) behavior policy, inserts
  trajectories into a replay buffer, and periodically updates its policy by
  sampling these transitions using prioritization.
  """

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      networks: r2d2_networks.R2D2Networks,
      config: r2d2_config.R2D2Config,
      seed: int,
      workdir: Optional[str] = '~/acme',
      counter: Optional[counting.Counter] = None,
  ):
    super().__init__(
        seed=seed,
        environment_spec=spec,
        builder=builder.R2D2Builder(config),
        networks=networks,
        policy_network=r2d2_actor.make_behavior_policy(networks, config),
        workdir=workdir,
        batch_size=config.batch_size,
        num_sgd_steps_per_step=config.sequence_period,
        counter=counter,
    )
