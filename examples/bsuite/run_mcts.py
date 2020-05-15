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

"""Example running MCTS on BSuite in a single process."""

from typing import Tuple

from absl import app
from absl import flags

import acme
from acme import networks
from acme import specs
from acme import wrappers
from acme.agents import mcts
from acme.agents.mcts import models
from acme.agents.mcts.models import mlp
from acme.agents.mcts.models import simulator

import bsuite
import dm_env
import sonnet as snt

flags.DEFINE_boolean('simulator', True, 'Simulator or learned model?')
FLAGS = flags.FLAGS


def make_env_and_model() -> Tuple[dm_env.Environment, models.Model]:
  """Create environment and corresponding model (learned or simulator)."""
  environment = bsuite.load('catch', kwargs={})
  if FLAGS.simulator:
    model = simulator.Simulator(environment)  # pytype: disable=attribute-error
  else:
    model = mlp.MLPModel(
        specs.make_environment_spec(environment),
        replay_capacity=1000,
        batch_size=16,
        hidden_sizes=(50,),
        embedding_size=50)
  environment = wrappers.SinglePrecisionWrapper(environment)

  return environment, model


def make_network(action_spec: specs.DiscreteArray) -> snt.Module:
  return snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([50, 50]),
      networks.PolicyValueHead(action_spec.num_values),
  ])


def main(_):
  # Create an environment and environment model.
  environment, model = make_env_and_model()
  environment_spec = specs.make_environment_spec(environment)

  # Create the network and optimizer.
  network = make_network(environment_spec.actions)
  optimizer = snt.optimizers.Adam(learning_rate=1e-3)

  # Construct the agent.
  agent = mcts.MCTS(
      environment_spec=environment_spec,
      model=model,
      network=network,
      optimizer=optimizer,
      discount=0.99,
      replay_capacity=10000,
      n_step=1,
      batch_size=16,
      num_simulations=50,
  )

  # Run the environment loop.
  loop = acme.EnvironmentLoop(environment, agent)
  loop.run(num_episodes=environment.bsuite_num_episodes)  # pytype: disable=attribute-error


if __name__ == '__main__':
  app.run(main)
