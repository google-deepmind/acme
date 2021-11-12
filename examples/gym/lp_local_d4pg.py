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

"""Example running D4PG on the control suite.

It runs the distributed agent using Launchpad runtime specified by
--lp_launch_type flag.
"""

from absl import app
from absl import flags
from acme.agents.tf import d4pg
import helpers
from acme.utils import lp_utils

import launchpad as lp

FLAGS = flags.FLAGS
flags.DEFINE_string('task', 'MountainCarContinuous-v0', 'Gym task name (str).')


def main(_):
  environment_factory = lp_utils.partial_kwargs(
      helpers.make_environment, task=FLAGS.task)

  program = d4pg.DistributedD4PG(
      environment_factory=environment_factory,
      network_factory=lp_utils.partial_kwargs(helpers.make_networks),
      num_actors=2).build()

  lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))


if __name__ == '__main__':
  app.run(main)
