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

"""Example running DQN on OpenSpiel in a single process."""

from absl import app
from absl import flags

import acme
from acme import specs
from acme import wrappers
from acme.open_spiel import open_spiel_environment_loop
from acme.open_spiel import open_spiel_specs
from acme.open_spiel import open_spiel_wrapper
from acme.open_spiel.agents.tf import dqn
from acme.tf import networks
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
  environment_spec = specs.make_environment_spec(environment)
  extras_spec = open_spiel_specs.make_extras_spec(environment)

  network = snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([50, 50, environment_spec.actions.num_values])
  ])

  # Construct the agent.
  agents = []

  for i in range(environment.num_players):
    agents.append(
        dqn.DQN(
            environment_spec=environment_spec,
            extras_spec=extras_spec,
            priority_exponent=0.0,  # TODO Test priority_exponent.
            discount=1.0,
            n_step=1,  # TODO Appear to be convergence issues when n > 1.
            epsilon=0.1,
            network=network,
            player_id=i))

  # Run the environment loop.
  loop = open_spiel_environment_loop.OpenSpielEnvironmentLoop(
      environment, agents)
  loop.run(num_episodes=100000)  # pytype: disable=attribute-error


if __name__ == '__main__':
  app.run(main)
