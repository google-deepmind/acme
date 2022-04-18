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

"""Example running DQfD on BSuite in a single process.
"""

from absl import app
from absl import flags

import acme
from acme import specs
from acme import wrappers
from acme.agents.tf import dqfd
from acme.agents.tf.dqfd import bsuite_demonstrations

import bsuite
import sonnet as snt


# Bsuite flags
flags.DEFINE_string('bsuite_id', 'deep_sea/0', 'Bsuite id.')
flags.DEFINE_string('results_dir', '/tmp/bsuite', 'CSV results directory.')
flags.DEFINE_boolean('overwrite', False, 'Whether to overwrite csv results.')

# Agent flags
flags.DEFINE_float('demonstration_ratio', 0.5,
                   ('Proportion of demonstration transitions in the replay '
                    'buffer.'))
flags.DEFINE_integer('n_step', 5,
                     ('Number of steps to squash into a single transition.'))
flags.DEFINE_float('samples_per_insert', 8,
                   ('Number of samples to take from replay for every insert '
                    'that is made.'))
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')

FLAGS = flags.FLAGS


def make_network(action_spec: specs.DiscreteArray) -> snt.Module:
  return snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([50, 50, action_spec.num_values]),
  ])


def main(_):
  # Create an environment and grab the spec.
  raw_environment = bsuite.load_and_record_to_csv(
      bsuite_id=FLAGS.bsuite_id,
      results_dir=FLAGS.results_dir,
      overwrite=FLAGS.overwrite,
  )
  environment = wrappers.SinglePrecisionWrapper(raw_environment)
  environment_spec = specs.make_environment_spec(environment)

  # Construct the agent.
  agent = dqfd.DQfD(
      environment_spec=environment_spec,
      network=make_network(environment_spec.actions),
      demonstration_dataset=bsuite_demonstrations.make_dataset(
          raw_environment, stochastic=False),
      demonstration_ratio=FLAGS.demonstration_ratio,
      samples_per_insert=FLAGS.samples_per_insert,
      learning_rate=FLAGS.learning_rate)

  # Run the environment loop.
  loop = acme.EnvironmentLoop(environment, agent)
  loop.run(num_episodes=environment.bsuite_num_episodes)  # pytype: disable=attribute-error


if __name__ == '__main__':
  app.run(main)
