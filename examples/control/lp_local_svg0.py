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

"""Example running SVG0 on the control suite."""

from absl import app
from absl import flags
from acme.agents.tf import svg0_prior
import helpers

from acme.utils import lp_utils

import launchpad as lp

FLAGS = flags.FLAGS
flags.DEFINE_string('domain', 'cartpole', 'Control suite domain name (str).')
flags.DEFINE_string('task', 'balance', 'Control suite task name (str).')


def main(_):
  environment_factory = lp_utils.partial_kwargs(
      helpers.make_environment, domain_name=FLAGS.domain, task_name=FLAGS.task)

  batch_size = 32
  sequence_length = 20
  gradient_steps_per_actor_step = 1.0
  samples_per_insert = (
      gradient_steps_per_actor_step * batch_size * sequence_length)
  num_actors = 1

  program = svg0_prior.DistributedSVG0(
      environment_factory=environment_factory,
      network_factory=lp_utils.partial_kwargs(svg0_prior.make_default_networks),
      batch_size=batch_size,
      sequence_length=sequence_length,
      samples_per_insert=samples_per_insert,
      entropy_regularizer_cost=1e-4,
      max_replay_size=int(2e6),
      target_update_period=250,
      num_actors=num_actors).build()

  lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING)


if __name__ == '__main__':
  app.run(main)
