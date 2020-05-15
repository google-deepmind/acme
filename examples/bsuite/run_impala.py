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

"""Runs IMPALA on bsuite locally."""

from absl import app

import acme
from acme import networks
from acme import specs
from acme import wrappers
from acme.agents import impala

import bsuite
import dm_env
import sonnet as snt


def make_environment() -> dm_env.Environment:
  environment = bsuite.load('catch', kwargs={})
  environment = wrappers.SinglePrecisionWrapper(environment)
  return environment


def make_network(action_spec: specs.DiscreteArray) -> snt.RNNCore:
  return snt.DeepRNN([
      snt.Flatten(),
      snt.nets.MLP([50, 50]),
      snt.LSTM(20),
      networks.PolicyValueHead(action_spec.num_values),
  ])


def main(_):
  # Create an environment and grab the spec.
  environment = make_environment()
  environment_spec = specs.make_environment_spec(environment)

  # Create the networks to optimize.
  network = make_network(environment_spec.actions)

  agent = impala.IMPALA(
      environment_spec=environment_spec,
      network=network,
      sequence_length=3,
      sequence_period=3,
  )

  # Run the environment loop.
  loop = acme.EnvironmentLoop(environment, agent)
  loop.run(num_episodes=environment.bsuite_num_episodes)  # pytype: disable=attribute-error


if __name__ == '__main__':
  app.run(main)
