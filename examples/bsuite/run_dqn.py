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

"""Example running DQN on BSuite in a single process."""

from absl import app

import acme
from acme import specs
from acme import wrappers
from acme.agents import dqn

import bsuite
import sonnet as snt


def main(_):
  # Create an environment and grab the spec.
  environment = bsuite.load_from_id('catch/0')
  environment = wrappers.SinglePrecisionWrapper(environment)
  environment_spec = specs.make_environment_spec(environment)

  network = snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([50, 50, environment_spec.actions.num_values])
  ])

  # Construct the agent.
  agent = dqn.DQN(
      environment_spec=environment_spec, network=network)

  # Run the environment loop.
  loop = acme.EnvironmentLoop(environment, agent)
  loop.run(num_episodes=environment.bsuite_num_episodes)  # pytype: disable=attribute-error


if __name__ == '__main__':
  app.run(main)
