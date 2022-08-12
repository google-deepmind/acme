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

"""Defines distributed and local multiagent decentralized agents."""

import functools
from typing import Any, Dict, Optional, Sequence, Tuple

from acme import specs
from acme.agents.jax.multiagent.decentralized import builder as decentralized_builders
from acme.agents.jax.multiagent.decentralized import config as decentralized_config
from acme.agents.jax.multiagent.decentralized import factories as decentralized_factories
from acme.jax import types
from acme.jax import utils
from acme.jax.layouts import distributed_layout
from acme.jax.layouts import local_layout
from acme.multiagent import types as ma_types
from acme.utils import counting


class DistributedDecentralizedMultiAgent(distributed_layout.DistributedLayout):
  """Distributed program definition for decentralized multiagent learning."""

  def __init__(
      self,
      agent_types: Dict[ma_types.AgentID, ma_types.GenericAgent],
      environment_factory: types.EnvironmentFactory,
      network_factory: ma_types.NetworkFactory,
      policy_factory: ma_types.PolicyFactory,
      builder_factory: ma_types.BuilderFactory,
      config: decentralized_config.DecentralizedMultiagentConfig,
      seed: int,
      num_parallel_actors_per_agent: int,
      environment_spec: Optional[specs.EnvironmentSpec] = None,
      max_number_of_steps: Optional[int] = None,
      log_to_bigtable: bool = False,
      log_every: float = 10.0,
      evaluator_factories: Optional[Sequence[
          distributed_layout.EvaluatorFactory]] = None,
  ):
    assert len(set(agent_types.values())) == 1, (
        f'Sub-agent types must be identical, but are {agent_types}.')

    learner_logger_fns = decentralized_factories.default_logger_factory(
        agent_types=agent_types,
        base_label='learner',
        save_data=log_to_bigtable,
        time_delta=log_every,
        asynchronous=True,
        serialize_fn=utils.fetch_devicearray,
        steps_key='learner_steps')
    builders = builder_factory(agent_types, config.sub_agent_configs)

    train_policy_factory = functools.partial(policy_factory, eval_mode=False)
    if evaluator_factories is None:
      eval_network_fn = functools.partial(policy_factory, eval_mode=True)
      evaluator_factories = [
          distributed_layout.default_evaluator_factory(
              environment_factory=environment_factory,
              network_factory=network_factory,
              policy_factory=eval_network_fn,
              save_logs=log_to_bigtable)
      ]
    self.builder = decentralized_builders.DecentralizedMultiAgentBuilder(
        builders)
    # pytype: disable=wrong-arg-types
    super().__init__(
        seed=seed,
        environment_factory=environment_factory,
        network_factory=network_factory,
        builder=self.builder,
        policy_network=train_policy_factory,
        evaluator_factories=evaluator_factories,
        num_actors=num_parallel_actors_per_agent,
        environment_spec=environment_spec,
        max_number_of_steps=max_number_of_steps,
        prefetch_size=config.prefetch_size,
        save_logs=log_to_bigtable,
        actor_logger_fn=distributed_layout.get_default_logger_fn(
            log_to_bigtable, log_every),
        learner_logger_fn=learner_logger_fns
    )
    # pytype: enable=wrong-arg-types


class DecentralizedMultiAgent(local_layout.LocalLayout):
  """Local definition for decentralized multiagent learning."""

  def __init__(
      self,
      agent_types: Dict[ma_types.AgentID, ma_types.GenericAgent],
      spec: specs.EnvironmentSpec,
      builder_factory: ma_types.BuilderFactory,
      networks: ma_types.MultiAgentNetworks,
      policy_networks: ma_types.MultiAgentPolicyNetworks,
      config: decentralized_config.DecentralizedMultiagentConfig,
      seed: int,
      workdir: Optional[str] = '~/acme',
      counter: Optional[counting.Counter] = None,
      save_data: bool = True
  ):
    assert len(set(agent_types.values())) == 1, (
        f'Sub-agent types must be identical, but are {agent_types}.')
    # TODO(somidshafiei): add input normalizer. However, this may require
    # adding some helper utilities for each single-agent algorithms, as
    # batch_dims  for NormalizationBuilder are algorithm-dependent (e.g., see
    # PPO vs. SAC JAX agents)

    learner_logger_fns = decentralized_factories.default_logger_factory(
        agent_types=agent_types,
        base_label='learner',
        save_data=save_data,
        steps_key='learner_steps')
    learner_loggers = {agent_id: learner_logger_fns[agent_id]()
                       for agent_id in agent_types.keys()}
    builders = builder_factory(agent_types, config.sub_agent_configs)
    self.builder = decentralized_builders.DecentralizedMultiAgentBuilder(
        builders)
    # pytype: disable=wrong-arg-types
    super().__init__(
        seed=seed,
        environment_spec=spec,
        builder=self.builder,
        networks=networks,
        policy_network=policy_networks,
        prefetch_size=config.prefetch_size,
        learner_logger=learner_loggers,
        batch_size=config.batch_size,
        workdir=workdir,
        counter=counter,
    )
    # pytype: enable=wrong-arg-types


def init_decentralized_multiagent(
    agent_types: Dict[ma_types.AgentID, ma_types.GenericAgent],
    environment_spec: specs.EnvironmentSpec,
    seed: int,
    batch_size: int,
    workdir: Optional[str] = '~/acme',
    init_network_fn: Optional[ma_types.InitNetworkFn] = None,
    init_policy_network_fn: Optional[ma_types.InitPolicyNetworkFn] = None,
    save_data: bool = True,
    config_overrides: Optional[Dict[ma_types.AgentID, Dict[str, Any]]] = None
    ) -> Tuple[DecentralizedMultiAgent, ma_types.MultiAgentPolicyNetworks]:
  """Returns decentralized multiagent LocalLayout instance.

  Intended to be used as a helper function to more readily instantiate and
  experiment with multiagent setups. For full functionality, use
  DecentralizedMultiAgent or DistributedDecentralizedMultiAgent directly.

  Args:
    agent_types: a dict specifying the agent identifier and their types
      (e.g., {'0': factories.DefaultSupportedAgents.PPO, '1': ...}).
    environment_spec: environment spec.
    seed: seed.
    batch_size: the batch size (used for each sub-agent).
    workdir: working directory (e.g., used for checkpointing).
    init_network_fn: optional custom network initializer function.
    init_policy_network_fn: optional custom policy network initializer function.
    save_data: whether to save data throughout training.
    config_overrides: a dict specifying agent-specific configuration overrides.
  """
  configs = decentralized_factories.default_config_factory(
      agent_types, batch_size, config_overrides)
  networks = decentralized_factories.network_factory(environment_spec,
                                                     agent_types,
                                                     init_network_fn)
  policy_networks = decentralized_factories.policy_network_factory(
      networks, environment_spec, agent_types, configs, eval_mode=False,
      init_policy_network_fn=init_policy_network_fn)
  eval_policy_networks = decentralized_factories.policy_network_factory(
      networks, environment_spec, agent_types, configs, eval_mode=True,
      init_policy_network_fn=init_policy_network_fn)
  config = decentralized_config.DecentralizedMultiagentConfig(
      batch_size=batch_size, sub_agent_configs=configs)
  decentralized_multi_agent = DecentralizedMultiAgent(
      agent_types=agent_types,
      spec=environment_spec,
      builder_factory=decentralized_factories.builder_factory,
      networks=networks,
      policy_networks=policy_networks,
      seed=seed,
      config=config,
      workdir=workdir,
      save_data=save_data
  )
  return decentralized_multi_agent, eval_policy_networks
