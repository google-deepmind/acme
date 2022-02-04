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

"""Example running PPO on the OpenAI Gym.

Runs the synchronous PPO agent.
"""

from absl import flags
import acme
from acme import specs
from acme.agents.jax import ppo
from absl import app
import helpers
from acme.utils import counting
from acme.utils import experiment_utils
import jax

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_steps', 4000000,
                     'Number of env steps to run training for.')
flags.DEFINE_integer('eval_every', 10000, 'How often to run evaluation')
flags.DEFINE_string('env_name', 'MountainCarContinuous-v0',
                    'What environment to run')

# PPO agent configuration flags.
flags.DEFINE_integer('unroll_length', 16, 'Unroll length.')
flags.DEFINE_integer('num_minibatches', 32, 'Number of minibatches.')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 2048 // 16, 'Batch size.')

flags.DEFINE_integer('seed', 0, 'Random seed.')


def main(_):
  # Create an environment, grab the spec, and use it to create networks.
  environment = helpers.make_environment(task=FLAGS.env_name)
  environment_spec = specs.make_environment_spec(environment)
  agent_networks = ppo.make_gym_networks(environment_spec)

  # Construct the agent.
  config = ppo.PPOConfig(
      unroll_length=FLAGS.unroll_length,
      num_minibatches=FLAGS.num_minibatches,
      num_epochs=FLAGS.num_epochs,
      batch_size=FLAGS.batch_size)

  learner_logger = experiment_utils.make_experiment_logger(
      label='learner', steps_key='learner_steps')
  agent = ppo.PPO(
      environment_spec,
      agent_networks,
      config=config,
      seed=FLAGS.seed,
      counter=counting.Counter(prefix='learner'),
      logger=learner_logger)

  # Create the environment loop used for training.
  train_logger = experiment_utils.make_experiment_logger(
      label='train', steps_key='train_steps')
  train_loop = acme.EnvironmentLoop(
      environment,
      agent,
      counter=counting.Counter(prefix='train'),
      logger=train_logger)

  # Create the evaluation actor and loop.
  eval_logger = experiment_utils.make_experiment_logger(
      label='eval', steps_key='eval_steps')
  eval_actor = agent.builder.make_actor(
      random_key=jax.random.PRNGKey(FLAGS.seed),
      policy_network=ppo.make_inference_fn(agent_networks, evaluation=True),
      variable_source=agent)
  eval_env = helpers.make_environment(task=FLAGS.env_name)
  eval_loop = acme.EnvironmentLoop(
      eval_env,
      eval_actor,
      counter=counting.Counter(prefix='eval'),
      logger=eval_logger)

  assert FLAGS.num_steps % FLAGS.eval_every == 0
  for _ in range(FLAGS.num_steps // FLAGS.eval_every):
    eval_loop.run(num_episodes=5)
    train_loop.run(num_steps=FLAGS.eval_every)
  eval_loop.run(num_episodes=5)

if __name__ == '__main__':
  app.run(main)
