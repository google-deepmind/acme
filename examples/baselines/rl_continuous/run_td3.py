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

"""Example running SAC on continuous control tasks."""

from absl import flags
from acme import specs
from acme.agents.jax import td3
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

  environment_spec = specs.make_environment_spec(environment)
  agent_networks = td3.make_networks(
      environment_spec, hidden_layer_sizes=(256, 256, 256))

  # Construct the agent.
  config = td3.TD3Config(
      policy_learning_rate=3e-4,
      critic_learning_rate=3e-4,
  )
  learner_logger = experiment_utils.make_experiment_logger(
      label='learner', steps_key='learner_steps')

  td3_builder = td3.TD3Builder(config, logger_fn=(lambda: learner_logger))
  # pylint:disable=g-long-lambda
  experiment = experiments.Config(
      builder=td3_builder,
      environment_factory=lambda seed: environment,
      network_factory=lambda spec: agent_networks,
      policy_network_factory=(lambda network: td3.get_default_behavior_policy(
          networks=network,
          action_specs=environment_spec.actions,
          sigma=config.sigma)),
      eval_policy_network_factory=(
          lambda network: td3.get_default_behavior_policy(
              network, environment_spec.actions, sigma=0)),
      seed=FLAGS.seed,
      max_number_of_steps=FLAGS.num_steps)
  # pylint:enable=g-long-lambda
  experiments.run_experiment(
      experiment=experiment, eval_every=FLAGS.eval_every, num_eval_episodes=10)


if __name__ == '__main__':
  app.run(main)
