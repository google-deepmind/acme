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

"""Run DQN on Atari."""

from absl import app
from absl import flags
import acme
from acme.agents.tf import dqn
import helpers
from acme.tf import networks

flags.DEFINE_string('level', 'PongNoFrameskip-v4', 'Which Atari level to play.')
flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes to train for.')

FLAGS = flags.FLAGS


def main(_):
  env = helpers.make_environment(FLAGS.level)
  env_spec = acme.make_environment_spec(env)
  network = networks.DQNAtariNetwork(env_spec.actions.num_values)

  agent = dqn.DQN(env_spec, network)

  loop = acme.EnvironmentLoop(env, agent)
  loop.run(FLAGS.num_episodes)


if __name__ == '__main__':
  app.run(main)
