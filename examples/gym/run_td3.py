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

"""Example running TD3 on the control suite."""

from absl import flags
from acme import specs
from acme.agents.jax import td3
from absl import app
import helpers
from acme.jax import experiments
from acme.utils import experiment_utils

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_steps', 1000000,
                     'Number of env steps to run training for.')
flags.DEFINE_integer('eval_every', 10000, 'How often to run evaluation')
flags.DEFINE_string('env_name', 'MountainCarContinuous-v0',
                    'What environment to run')
flags.DEFINE_integer('num_sgd_steps_per_step', 1,
                     'Number of SGD steps per learner step().')
flags.DEFINE_integer('seed', 0, 'Random seed.')


def main(_):
  # Create an environment, grab the spec, and use it to create networks.
  environment = helpers.make_environment(task=FLAGS.env_name)
  environment_spec = specs.make_environment_spec(environment)
  agent_networks = td3.make_networks(environment_spec)

  config = td3.TD3Config(num_sgd_steps_per_step=FLAGS.num_sgd_steps_per_step)
  learner_logger = experiment_utils.make_experiment_logger(
      label='learner', steps_key='learner_steps')

  td3_builder = td3.TD3Builder(config, logger_fn=(lambda: learner_logger))
  policy_network = td3.get_default_behavior_policy(
      networks=agent_networks,
      action_specs=environment_spec.actions,
      sigma=config.sigma)
  eval_policy_network = td3.get_default_behavior_policy(
      agent_networks, environment_spec.actions, sigma=0.)
  experiment = experiments.Config(
      builder=td3_builder,
      environment_factory=lambda seed: environment,
      network_factory=lambda spec: agent_networks,
      policy_network_factory=lambda networks: policy_network,
      eval_policy_network_factory=lambda network: eval_policy_network,
      seed=FLAGS.seed,
      max_number_of_steps=FLAGS.num_steps)

  experiments.run_experiment(experiment=experiment, eval_every=FLAGS.eval_every)


if __name__ == '__main__':
  app.run(main)
