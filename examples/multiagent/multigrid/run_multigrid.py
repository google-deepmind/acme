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
from absl import flags

import acme
from acme import specs
from acme.agents.jax.multiagent.decentralized import agents
from acme.agents.jax.multiagent.decentralized import factories
from absl import app
import helpers
from acme.utils import loggers
from acme.wrappers import multigrid_wrapper
import jax

FLAGS = flags.FLAGS
_NUM_STEPS = flags.DEFINE_integer('num_steps', 10000,
                                  'Number of env steps to run training for.')
_EVAL_EVERY = flags.DEFINE_integer('eval_every', 1000,
                                   'How often to run evaluation.')
_ENV_NAME = flags.DEFINE_string('env_name', 'MultiGrid-Empty-5x5-v0',
                                'What environment to run.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 64, 'Batch size.')
_SEED = flags.DEFINE_integer('seed', 0, 'Random seed.')


def main(_):
  """Runs multigrid experiment."""
  # Training environment
  train_env = multigrid_wrapper.make_multigrid_environment(_ENV_NAME.value)
  train_environment_spec = specs.make_environment_spec(train_env)

  agent_types = {
      str(i): factories.DefaultSupportedAgent.PPO
      for i in range(train_env.num_agents)  # pytype: disable=attribute-error
  }
  # Example of how to set custom sub-agent configurations.
  ppo_configs = {'unroll_length': 16, 'num_minibatches': 32, 'num_epochs': 10}
  config_overrides = {
      k: ppo_configs for k, v in agent_types.items() if v == 'ppo'
  }
  train_agents, eval_policy_networks = agents.init_decentralized_multiagent(
      agent_types=agent_types,
      environment_spec=train_environment_spec,
      seed=_SEED.value,
      batch_size=_BATCH_SIZE.value,
      init_network_fn=helpers.init_default_multigrid_network,
      config_overrides=config_overrides
      )

  train_loop = acme.EnvironmentLoop(
      train_env,
      train_agents,
      label='train_loop',
      logger=loggers.TerminalLogger(
          label='trainer', time_delta=1.0))

  # Evaluation environment
  eval_env = multigrid_wrapper.make_multigrid_environment(_ENV_NAME.value)
  eval_environment_spec = specs.make_environment_spec(eval_env)
  eval_actors = train_agents.builder.make_actor(
      random_key=jax.random.PRNGKey(_SEED.value),
      policy_networks=eval_policy_networks,
      environment_spec=eval_environment_spec,
      variable_source=train_agents
      )
  eval_loop = acme.EnvironmentLoop(
      eval_env,
      eval_actors,
      label='eval_loop',
      logger=loggers.TerminalLogger(
          label='evaluator', time_delta=1.0))

  # Run
  assert _NUM_STEPS.value % _EVAL_EVERY.value == 0
  for _ in range(_NUM_STEPS.value // _EVAL_EVERY.value):
    eval_loop.run(num_episodes=5)
    train_loop.run(num_steps=_EVAL_EVERY.value)
  eval_loop.run(num_episodes=5)

  return train_agents

if __name__ == '__main__':
  app.run(main)
