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

"""Example running D4PG on continuous control tasks."""

from absl import flags
from acme import specs
from acme.agents.jax import d4pg
import helpers
from absl import app
from acme.jax import experiments
from acme.utils import experiment_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'gym:HalfCheetah-v2', 'What environment to run')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_steps', 1_000_000, 'Number of env steps to run.')
flags.DEFINE_integer('eval_every', 50_000, 'How often to run evaluation.')


def main(_):
  # Create an environment, grab the spec, and use it to create networks.

  suite, task = FLAGS.env_name.split(':', 1)
  environment = helpers.make_environment(suite, task)

  # Bound of the distributional critic. The reward for control environments is
  # normalized, not for gym locomotion environments hence the different scales.
  vmax_values = {
      'gym': 1000.,
      'control': 150.,
  }
  vmax = vmax_values[suite]

  environment_spec = specs.make_environment_spec(environment)
  agent_networks = d4pg.make_networks(
      environment_spec,
      policy_layer_sizes=(256, 256, 256),
      critic_layer_sizes=(256, 256, 256),
      vmin=-vmax, vmax=vmax,)

  # Construct the agent.
  config = d4pg.D4PGConfig(
      learning_rate=3e-4,
      sigma=0.2
  )

  learner_logger = experiment_utils.make_experiment_logger(
      label='learner', steps_key='learner_steps')

  d4pg_builder = d4pg.D4PGBuilder(config, logger_fn=(lambda: learner_logger))

  experiment = experiments.Config(
      builder=d4pg_builder,
      environment_factory=lambda seed: environment,
      network_factory=lambda spec: agent_networks,
      policy_network_factory=(
          lambda network: d4pg.get_default_behavior_policy(network, config)),
      eval_policy_network_factory=d4pg.get_default_eval_policy,
      seed=FLAGS.seed,
      max_number_of_steps=FLAGS.num_steps)
  experiments.run_experiment(
      experiment=experiment, eval_every=FLAGS.eval_every, num_eval_episodes=10)


if __name__ == '__main__':
  app.run(main)
