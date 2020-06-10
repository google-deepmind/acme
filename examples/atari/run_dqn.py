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

import functools

from absl import app
from absl import flags
import acme
from acme import wrappers
from acme.agents.tf import dqn
from acme.tf import networks
import dm_env
import gym

flags.DEFINE_string('level', 'PongNoFrameskip-v4', 'Which Atari level to play.')
flags.DEFINE_integer('num_episodes', 1000, 'Number of episodes to train for.')

FLAGS = flags.FLAGS


def make_environment(evaluation: bool = False) -> dm_env.Environment:

  env = gym.make(FLAGS.level, full_action_space=True)

  max_episode_len = 108_000 if evaluation else 50_000

  return wrappers.wrap_all(env, [
      wrappers.GymAtariAdapter,
      functools.partial(
          wrappers.AtariWrapper,
          to_float=True,
          max_episode_len=max_episode_len,
          zero_discount_on_life_loss=True,
      ),
      wrappers.SinglePrecisionWrapper,
  ])


def main(_):
  env = make_environment()
  env_spec = acme.make_environment_spec(env)
  network = networks.DQNAtariNetwork(env_spec.actions.num_values)

  agent = dqn.DQN(env_spec, network)

  loop = acme.EnvironmentLoop(env, agent)
  loop.run(FLAGS.num_episodes)


if __name__ == '__main__':
  app.run(main)
