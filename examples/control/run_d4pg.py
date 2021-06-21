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

"""Example running D4PG on the control suite."""

from absl import app
from absl import flags
import acme
from acme import specs
from acme.agents.tf import actors
from acme.agents.tf import d4pg
import helpers
import sonnet as snt

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes', 100,
                     'Number of training episodes to run for.')

flags.DEFINE_integer('num_episodes_per_eval', 10,
                     'Number of training episodes to run between evaluation '
                     'episodes.')


def main(_):
  # Create an environment, grab the spec, and use it to create networks.
  environment = helpers.make_environment()
  environment_spec = specs.make_environment_spec(environment)
  agent_networks = d4pg.make_default_networks(environment_spec.actions)

  # Construct the agent.
  agent = d4pg.D4PG(
      environment_spec=environment_spec,
      policy_network=agent_networks['policy'],
      critic_network=agent_networks['critic'],
      observation_network=agent_networks['observation'],  # pytype: disable=wrong-arg-types
  )

  # Create the environment loop used for training.
  train_loop = acme.EnvironmentLoop(environment, agent, label='train_loop')

  # Create the evaluation policy.
  eval_policy = snt.Sequential([
      agent_networks['observation'],
      agent_networks['policy'],
  ])

  # Create the evaluation actor and loop.
  eval_actor = actors.FeedForwardActor(policy_network=eval_policy)
  eval_env = helpers.make_environment()
  eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label='eval_loop')

  for _ in range(FLAGS.num_episodes // FLAGS.num_episodes_per_eval):
    train_loop.run(num_episodes=FLAGS.num_episodes_per_eval)
    eval_loop.run(num_episodes=1)


if __name__ == '__main__':
  app.run(main)
