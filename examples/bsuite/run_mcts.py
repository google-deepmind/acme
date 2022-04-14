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
from acme import specs
from acme import wrappers
from acme.agents.tf import mcts
from acme.agents.tf.mcts import models
from acme.agents.tf.mcts.models import mlp
from acme.agents.tf.mcts.models import simulator
from acme.tf import networks
import bsuite
from bsuite.logging import csv_logging
import dm_env
import sonnet as snt

# Bsuite flags
flags.DEFINE_string('bsuite_id', 'deep_sea/0', 'Bsuite id.')
flags.DEFINE_string('results_dir', '/tmp/bsuite', 'CSV results directory.')
flags.DEFINE_boolean('overwrite', False, 'Whether to overwrite csv results.')
# Agent flags
flags.DEFINE_boolean('simulator', True, 'Simulator or learned model?')
FLAGS = flags.FLAGS


def make_env_and_model(
    bsuite_id: str,
    results_dir: str,
    overwrite: bool) -> Tuple[dm_env.Environment, models.Model]:
  """Create environment and corresponding model (learned or simulator)."""
  raw_env = bsuite.load_from_id(bsuite_id)
  if FLAGS.simulator:
    model = simulator.Simulator(raw_env)  # pytype: disable=attribute-error
  else:
    model = mlp.MLPModel(
        specs.make_environment_spec(raw_env),
        replay_capacity=1000,
        batch_size=16,
        hidden_sizes=(50,),
    )
  environment = csv_logging.wrap_environment(
      raw_env, bsuite_id, results_dir, overwrite)
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
  environment, model = make_env_and_model(
      bsuite_id=FLAGS.bsuite_id,
      results_dir=FLAGS.results_dir,
      overwrite=FLAGS.overwrite,
  )
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
