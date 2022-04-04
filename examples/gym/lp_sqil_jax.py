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

"""Example running SQIL+SAC in JAX on the OpenAI Gym.

It runs the distributed agent using Launchpad runtime specified by
--lp_launch_type flag.
"""

import functools

from absl import app
from absl import flags
from acme.agents.jax import sac
from acme.agents.jax import sqil
import helpers
from acme.utils import lp_utils
import launchpad as lp

FLAGS = flags.FLAGS
flags.DEFINE_string('task', 'MountainCarContinuous-v0',
                    'GYM environment task (str).')
flags.DEFINE_string(
    'dataset_name', 'd4rl_mujoco_halfcheetah/v0-medium', 'What dataset to use. '
    'See the TFDS catalog for possible values.')


def main(_):
  task = FLAGS.task
  environment_factory = lambda seed: helpers.make_environment(task)
  sac_config = sac.SACConfig(num_sgd_steps_per_step=64)
  sac_builder = sac.SACBuilder(sac_config)

  program = sqil.DistributedSQIL(
      environment_factory=environment_factory,
      rl_agent=sac_builder,
      network_factory=sac.make_networks,
      batch_size=sac_config.batch_size * sac_config.num_sgd_steps_per_step,
      make_demonstrations=functools.partial(
          helpers.make_demonstration_iterator, dataset_name=FLAGS.dataset_name),
      policy_network=sac.apply_policy_and_sample,
      evaluator_policy_network=(
          lambda n: sac.apply_policy_and_sample(n, eval_mode=True)),
      num_actors=4,
      max_number_of_steps=1000000).build()

  # Launch experiment.
  lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))


if __name__ == '__main__':
  app.run(main)
