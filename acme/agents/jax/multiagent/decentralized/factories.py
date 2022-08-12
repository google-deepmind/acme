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

"""Decentralized multiagent factories.

Used to unify agent initialization for both local and distributed layouts.
"""

import enum
import functools
from typing import Any, Callable, Dict, Mapping, Optional

from acme import specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import builders as jax_builders
from acme.agents.jax import ppo
from acme.agents.jax import sac
from acme.agents.jax import td3
from acme.multiagent import types as ma_types
from acme.multiagent import utils as ma_utils
from acme.utils import loggers


class DefaultSupportedAgent(enum.Enum):
  """Agents which have default initializers supported below."""
  TD3 = 'TD3'
  SAC = 'SAC'
  PPO = 'PPO'


def init_default_network(
    agent_type: DefaultSupportedAgent,
    agent_spec: specs.EnvironmentSpec) -> ma_types.Networks:
  """Returns default networks for a single agent."""
  if agent_type == DefaultSupportedAgent.TD3:
    return td3.make_networks(agent_spec)
  elif agent_type == DefaultSupportedAgent.SAC:
    return sac.make_networks(agent_spec)
  elif agent_type == DefaultSupportedAgent.PPO:
    return ppo.make_networks(agent_spec)
  else:
    raise ValueError(f'Unsupported agent type: {agent_type}.')


def init_default_policy_network(
    agent_type: DefaultSupportedAgent,
    network: ma_types.Networks,
    agent_spec: specs.EnvironmentSpec,
    config: ma_types.AgentConfig,
    eval_mode: ma_types.EvalMode = False) -> ma_types.PolicyNetwork:
  """Returns default policy network for a single agent."""
  if agent_type == DefaultSupportedAgent.TD3:
    sigma = 0. if eval_mode else config.sigma
    return td3.get_default_behavior_policy(
        network, agent_spec.actions, sigma=sigma)
  elif agent_type == DefaultSupportedAgent.SAC:
    return sac.apply_policy_and_sample(network, eval_mode=eval_mode)
  elif agent_type == DefaultSupportedAgent.PPO:
    return ppo.make_inference_fn(network, evaluation=eval_mode)
  else:
    raise ValueError(f'Unsupported agent type: {agent_type}.')


def init_default_builder(
    agent_type: DefaultSupportedAgent,
    agent_config: ma_types.AgentConfig,
) -> jax_builders.GenericActorLearnerBuilder:
  """Returns default builder for a single agent."""
  if agent_type == DefaultSupportedAgent.TD3:
    assert isinstance(agent_config, td3.TD3Config)
    return td3.TD3Builder(agent_config)
  elif agent_type == DefaultSupportedAgent.SAC:
    assert isinstance(agent_config, sac.SACConfig)
    return sac.SACBuilder(agent_config)
  elif agent_type == DefaultSupportedAgent.PPO:
    assert isinstance(agent_config, ppo.PPOConfig)
    return ppo.PPOBuilder(agent_config)
  else:
    raise ValueError(f'Unsupported agent type: {agent_type}.')


def init_default_config(
    agent_type: DefaultSupportedAgent,
    config_overrides: Dict[str, Any]) -> ma_types.AgentConfig:
  """Returns default config for a single agent."""
  if agent_type == DefaultSupportedAgent.TD3:
    return td3.TD3Config(**config_overrides)
  elif agent_type == DefaultSupportedAgent.SAC:
    return sac.SACConfig(**config_overrides)
  elif agent_type == DefaultSupportedAgent.PPO:
    return ppo.PPOConfig(**config_overrides)
  else:
    raise ValueError(f'Unsupported agent type: {agent_type}.')


def default_logger_factory(
    agent_types: Dict[ma_types.AgentID, DefaultSupportedAgent],
    base_label: str,
    save_data: bool,
    time_delta: float = 1.0,
    asynchronous: bool = False,
    print_fn: Optional[Callable[[str], None]] = None,
    serialize_fn: Optional[Callable[[Mapping[str, Any]], str]] = None,
    steps_key: str = 'steps',
) -> ma_types.MultiAgentLoggerFn:
  """Returns callable that constructs default logger for all agents."""
  logger_fns = {}
  for agent_id in agent_types.keys():
    logger_fns[agent_id] = functools.partial(
        loggers.make_default_logger,
        f'{base_label}{agent_id}',
        save_data=save_data,
        time_delta=time_delta,
        asynchronous=asynchronous,
        print_fn=print_fn,
        serialize_fn=serialize_fn,
        steps_key=steps_key,
    )
  return logger_fns


def default_config_factory(
    agent_types: Dict[ma_types.AgentID, DefaultSupportedAgent],
    batch_size: int,
    config_overrides: Optional[Dict[ma_types.AgentID, Dict[str, Any]]] = None
) -> Dict[ma_types.AgentID, ma_types.AgentConfig]:
  """Returns default configs for all agents.

  Args:
    agent_types: dict mapping agent IDs to their type.
    batch_size: shared batch size for all agents.
    config_overrides: dict mapping (potentially a subset of) agent IDs to their
      config overrides. This should include any mandatory config parameters for
      the agents that do not have default values.
  """
  configs = {}
  for agent_id, agent_type in agent_types.items():
    agent_config_overrides = dict(
        # batch_size is required by LocalLayout, which is shared amongst
        # the agents. Hence, we enforce a shared batch_size in builders.
        batch_size=batch_size,
        # Unique replay_table_name per agent.
        replay_table_name=f'{adders_reverb.DEFAULT_PRIORITY_TABLE}_agent{agent_id}'
    )
    if config_overrides is not None and agent_id in config_overrides:
      agent_config_overrides = {
          **config_overrides[agent_id],
          **agent_config_overrides  # Comes second to ensure batch_size override
      }
    configs[agent_id] = init_default_config(agent_type, agent_config_overrides)
  return configs


def network_factory(
    environment_spec: specs.EnvironmentSpec,
    agent_types: Dict[ma_types.AgentID, ma_types.GenericAgent],
    init_network_fn: Optional[ma_types.InitNetworkFn] = None
) -> ma_types.MultiAgentNetworks:
  """Returns networks for all agents.

  Args:
    environment_spec: environment spec.
    agent_types: dict mapping agent IDs to their type.
    init_network_fn: optional callable that handles the network initialization
      for all sub-agents. If this is not supplied, a default network initializer
      is used (if it is supported for the designated agent type).
  """
  init_fn = init_network_fn or init_default_network
  networks = {}
  for agent_id, agent_type in agent_types.items():
    single_agent_spec = ma_utils.get_agent_spec(environment_spec, agent_id)
    networks[agent_id] = init_fn(agent_type, single_agent_spec)
  return networks


def policy_network_factory(
    networks: ma_types.MultiAgentNetworks,
    environment_spec: specs.EnvironmentSpec,
    agent_types: Dict[ma_types.AgentID, ma_types.GenericAgent],
    agent_configs: Dict[ma_types.AgentID, ma_types.AgentConfig],
    eval_mode: ma_types.EvalMode,
    init_policy_network_fn: Optional[ma_types.InitPolicyNetworkFn] = None
) -> ma_types.MultiAgentPolicyNetworks:
  """Returns default policy networks for all agents.

  Args:
    networks: dict mapping agent IDs to their networks.
    environment_spec: environment spec.
    agent_types: dict mapping agent IDs to their type.
    agent_configs: dict mapping agent IDs to their config.
    eval_mode: whether the policy should be initialized in evaluation mode (only
      used if an init_policy_network_fn is not explicitly supplied).
    init_policy_network_fn: optional callable that handles the policy network
      initialization for all sub-agents. If this is not supplied, a default
      policy network initializer is used (if it is supported for the designated
      agent type).
  """
  init_fn = init_policy_network_fn or init_default_policy_network
  policy_networks = {}
  for agent_id, agent_type in agent_types.items():
    single_agent_spec = ma_utils.get_agent_spec(environment_spec, agent_id)
    policy_networks[agent_id] = init_fn(agent_type, networks[agent_id],
                                        single_agent_spec,
                                        agent_configs[agent_id], eval_mode)
  return policy_networks


def builder_factory(
    agent_types: Dict[ma_types.AgentID, ma_types.GenericAgent],
    agent_configs: Dict[ma_types.AgentID, ma_types.AgentConfig],
    init_builder_fn: Optional[ma_types.InitBuilderFn] = None
) -> Dict[ma_types.AgentID, jax_builders.GenericActorLearnerBuilder]:
  """Returns default policy networks for all agents."""
  init_fn = init_builder_fn or init_default_builder
  builders = {}
  for agent_id, agent_type in agent_types.items():
    builders[agent_id] = init_fn(agent_type, agent_configs[agent_id])
  return builders
