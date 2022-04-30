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
from acme.jax import runners

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
  td3_builder = td3.TD3Builder(config)
  policy_network = td3.get_default_behavior_policy(
      networks=agent_networks,
      action_specs=environment_spec.actions,
      sigma=config.sigma)
  eval_policy_network = td3.get_default_behavior_policy(
      agent_networks, environment_spec.actions, sigma=0.)
  runners.run_agent(
      builder=td3_builder,
      environment=environment,
      num_steps=FLAGS.num_steps,
      eval_every=FLAGS.eval_every,
      seed=FLAGS.seed,
      networks=agent_networks,
      policy_network=policy_network,
      eval_policy_network=eval_policy_network)


if __name__ == '__main__':
  app.run(main)
