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

"""Example running DQN on BSuite in a single process."""

import bsuite
import sonnet as snt
from absl import app, flags

import acme
from acme import specs, wrappers
from acme.agents.tf import dqn

# Bsuite flags
flags.DEFINE_string("bsuite_id", "deep_sea/0", "Bsuite id.")
flags.DEFINE_string("results_dir", "/tmp/bsuite", "CSV results directory.")
flags.DEFINE_boolean("overwrite", False, "Whether to overwrite csv results.")
FLAGS = flags.FLAGS


def main(_):
    # Create an environment and grab the spec.
    raw_environment = bsuite.load_and_record_to_csv(
        bsuite_id=FLAGS.bsuite_id,
        results_dir=FLAGS.results_dir,
        overwrite=FLAGS.overwrite,
    )
    environment = wrappers.SinglePrecisionWrapper(raw_environment)
    environment_spec = specs.make_environment_spec(environment)

    network = snt.Sequential(
        [snt.Flatten(), snt.nets.MLP([50, 50, environment_spec.actions.num_values])]
    )

    # Construct the agent.
    agent = dqn.DQN(environment_spec=environment_spec, network=network)

    # Run the environment loop.
    loop = acme.EnvironmentLoop(environment, agent)
    loop.run(
        num_episodes=environment.bsuite_num_episodes
    )  # pytype: disable=attribute-error


if __name__ == "__main__":
    app.run(main)
