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

"""Example running ARS in JAX on the OpenAI Gym.

It runs the distributed agent using Launchpad runtime specified by
--lp_launch_type flag.
"""

from absl import app
from absl import flags
from acme.agents.jax import ars
import helpers
from acme.utils import lp_utils
import launchpad as lp

FLAGS = flags.FLAGS
flags.DEFINE_string('task', 'Ant-v2', 'GYM environment task (str).')


def main(_):
  task = FLAGS.task
  environment_factory = lambda seed: helpers.make_environment(task)
  config = ars.ARSConfig()
  program = ars.DistributedARS(
      environment_factory=environment_factory,
      network_factory=ars.make_networks,
      config=config,
      seed=0,
      log_every=1,
      num_actors=4,
      max_number_of_steps=config.num_steps).build()

  # Launch experiment.
  lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))


if __name__ == '__main__':
  app.run(main)
