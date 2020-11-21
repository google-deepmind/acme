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

"""Example running DQN on OpenSpiel game in a single process."""

from absl import app
from absl import flags

import acme
from acme import wrappers
from acme.agents.tf import dqn
from acme.environment_loops import open_spiel_environment_loop
from acme.tf.networks import legal_actions
from acme.wrappers import open_spiel_wrapper
from open_spiel.python import rl_environment
import sonnet as snt

flags.DEFINE_string("game", "tic_tac_toe", "Name of the game")
flags.DEFINE_integer("num_players", None, "Number of players")

FLAGS = flags.FLAGS


def main(_):
  # Create an environment and grab the spec.
  env_configs = {"players": FLAGS.num_players} if FLAGS.num_players else {}
  raw_environment = rl_environment.Environment(FLAGS.game, **env_configs)

  environment = open_spiel_wrapper.OpenSpielWrapper(raw_environment)
  environment = wrappers.SinglePrecisionWrapper(environment)
  environment_spec = acme.make_environment_spec(environment)

  network = legal_actions.MaskedSequential([
      snt.Flatten(),
      snt.nets.MLP([50, 50, environment_spec.actions.num_values])
  ])

  policy_network = snt.Sequential(
      [network, legal_actions.EpsilonGreedy(epsilon=0.1, threshold=-1e8)])

  # Construct the agents.
  agents = []

  for i in range(environment.num_players):
    agents.append(
        dqn.DQN(
            environment_spec=environment_spec,
            discount=1.0,
            n_step=1,  # Note: does indeed converge for n > 1
            network=network,
            policy_network=policy_network))

  # Run the environment loop.
  loop = open_spiel_environment_loop.OpenSpielEnvironmentLoop(
      environment, agents)
  loop.run(num_episodes=100000)  # pytype: disable=attribute-error


if __name__ == '__main__':
  app.run(main)
