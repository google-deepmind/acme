# python3
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

"""Multiagent multigrid training run example."""
from typing import Callable, Dict

from absl import flags

from acme import specs
from acme.agents.jax.multiagent import decentralized
from absl import app
import helpers
from acme.jax import experiments
from acme.jax import types as jax_types
from acme.multiagent import types as ma_types
from acme.utils import lp_utils
from acme.wrappers import multigrid_wrapper
import dm_env
import launchpad as lp

FLAGS = flags.FLAGS
_RUN_DISTRIBUTED = flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
_NUM_STEPS = flags.DEFINE_integer('num_steps', 10000,
                                  'Number of env steps to run training for.')
_EVAL_EVERY = flags.DEFINE_integer('eval_every', 1000,
                                   'How often to run evaluation.')
_ENV_NAME = flags.DEFINE_string('env_name', 'MultiGrid-Empty-5x5-v0',
                                'What environment to run.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 64, 'Batch size.')
_SEED = flags.DEFINE_integer('seed', 0, 'Random seed.')


def _make_environment_factory(env_name: str) -> jax_types.EnvironmentFactory:

  def environment_factory(seed: int) -> dm_env.Environment:
    del seed
    return multigrid_wrapper.make_multigrid_environment(env_name)

  return environment_factory


def _make_network_factory(
    agent_types: Dict[ma_types.AgentID, ma_types.GenericAgent]
) -> Callable[[specs.EnvironmentSpec], ma_types.MultiAgentNetworks]:

  def environment_factory(
      environment_spec: specs.EnvironmentSpec) -> ma_types.MultiAgentNetworks:
    return decentralized.network_factory(environment_spec, agent_types,
                                         helpers.init_default_multigrid_network)

  return environment_factory


def build_experiment_config() -> experiments.ExperimentConfig[
    ma_types.MultiAgentNetworks, ma_types.MultiAgentPolicyNetworks,
    ma_types.MultiAgentSample]:
  """Returns a config for multigrid experiments."""

  environment_factory = _make_environment_factory(_ENV_NAME.value)
  environment = environment_factory(_SEED.value)
  agent_types = {
      str(i): decentralized.DefaultSupportedAgent.PPO
      for i in range(environment.num_agents)  # pytype: disable=attribute-error
  }
  # Example of how to set custom sub-agent configurations.
  ppo_configs = {'unroll_length': 16, 'num_minibatches': 32, 'num_epochs': 10}
  config_overrides = {
      k: ppo_configs for k, v in agent_types.items() if v == 'ppo'
  }

  configs = decentralized.default_config_factory(agent_types, _BATCH_SIZE.value,
                                                 config_overrides)

  builder = decentralized.DecentralizedMultiAgentBuilder(
      agent_types=agent_types, agent_configs=configs)

  return experiments.ExperimentConfig(
      builder=builder,
      environment_factory=environment_factory,
      network_factory=_make_network_factory(agent_types=agent_types),
      seed=_SEED.value,
      max_num_actor_steps=_NUM_STEPS.value)


def main(_):
  config = build_experiment_config()
  if _RUN_DISTRIBUTED.value:
    program = experiments.make_distributed_experiment(
        experiment=config, num_actors=4)
    lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
  else:
    experiments.run_experiment(
        experiment=config, eval_every=_EVAL_EVERY.value, num_eval_episodes=5)


if __name__ == '__main__':
  app.run(main)
