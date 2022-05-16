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

"""Example running SAC on continuous control tasks."""

from absl import flags
import acme
from acme import specs
from acme.agents.jax import td3
import helpers
from absl import app
from acme.utils import counting
from acme.utils import experiment_utils
import jax

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'gym:HalfCheetah-v2', 'What environment to run')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_steps', 1_000_000, 'Number of env steps to run.')
flags.DEFINE_integer('eval_every', 50_000, 'How often to run evaluation.')


def main(_):
  # Create an environment, grab the spec, and use it to create networks.

  suite, task = FLAGS.env_name.split(':', 1)
  environment = helpers.make_environment(suite, task)

  environment_spec = specs.make_environment_spec(environment)
  agent_networks = td3.make_networks(
      environment_spec, hidden_layer_sizes=(256, 256, 256))

  # Construct the agent.
  config = td3.TD3Config(
      policy_learning_rate=3e-4,
      critic_learning_rate=3e-4,
  )
  agent = td3.TD3(
      spec=environment_spec,
      network=agent_networks,
      config=config,
      seed=FLAGS.seed)

  # Create the environment loop used for training.
  train_logger = experiment_utils.make_experiment_logger(
      label='train', steps_key='train_steps')

  counter = counting.Counter()
  train_loop = acme.EnvironmentLoop(
      environment,
      agent,
      counter=counting.Counter(counter, prefix='train'),
      logger=train_logger)

  # Create the evaluation actor and loop.
  eval_logger = experiment_utils.make_experiment_logger(
      label='eval', steps_key='eval_steps')
  eval_actor = agent.builder.make_actor(
      random_key=jax.random.PRNGKey(FLAGS.seed),
      policy_network=td3.get_default_behavior_policy(
          agent_networks, environment_spec.actions, sigma=0),
      variable_source=agent)
  eval_env = helpers.make_environment(suite, task)
  eval_loop = acme.EnvironmentLoop(
      eval_env,
      eval_actor,
      counter=counting.Counter(counter, prefix='eval'),
      logger=eval_logger)

  assert FLAGS.num_steps % FLAGS.eval_every == 0
  for _ in range(int(FLAGS.num_steps // FLAGS.eval_every)):
    eval_loop.run(num_episodes=10)
    train_loop.run(num_steps=FLAGS.eval_every)
  eval_loop.run(num_episodes=10)


if __name__ == '__main__':
  app.run(main)
