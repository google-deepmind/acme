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

"""Example running SAC on the OpenAI Gym."""

from absl import flags
import acme
from acme import specs
from acme.agents.jax import sac
from absl import app
import helpers
import jax

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_steps', 1000000,
                     'Number of env steps to run training for.')
flags.DEFINE_integer('eval_every', 10000, 'How often to run evaluation')
flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'What environment to run')
flags.DEFINE_integer('num_sgd_steps_per_step', 1,
                     'Number of SGD steps per learner step().')
flags.DEFINE_integer('seed', 0, 'Random seed.')


def main(_):
  # Create an environment, grab the spec, and use it to create networks.
  environment = helpers.make_environment(task=FLAGS.env_name)
  environment_spec = specs.make_environment_spec(environment)
  agent_networks = sac.make_networks(environment_spec)

  # Construct the agent.
  config = sac.SACConfig(
      target_entropy=sac.target_entropy_from_env_spec(environment_spec),
      num_sgd_steps_per_step=FLAGS.num_sgd_steps_per_step)
  agent = sac.SAC(
      environment_spec, agent_networks, config=config, seed=FLAGS.seed)

  # Create the environment loop used for training.
  train_loop = acme.EnvironmentLoop(environment, agent, label='train_loop')
  # Create the evaluation actor and loop.
  eval_actor = agent.builder.make_actor(
      random_key=jax.random.PRNGKey(FLAGS.seed),
      policy_network=sac.apply_policy_and_sample(
          agent_networks, eval_mode=True),
      variable_source=agent)
  eval_env = helpers.make_environment(task=FLAGS.env_name)
  eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label='eval_loop')

  assert FLAGS.num_steps % FLAGS.eval_every == 0
  for _ in range(FLAGS.num_steps // FLAGS.eval_every):
    eval_loop.run(num_episodes=5)
    train_loop.run(num_steps=FLAGS.eval_every)
  eval_loop.run(num_episodes=5)

if __name__ == '__main__':
  app.run(main)
