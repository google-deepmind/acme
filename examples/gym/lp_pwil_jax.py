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

"""Example running PWIL+SAC in JAX on the OpenAI Gym.

It runs the distributed agent using Launchpad runtime specified by
--lp_launch_type flag.
"""

import functools

from absl import app
from absl import flags
from acme.agents.jax import pwil
from acme.agents.jax import sac
from acme.datasets import tfds
import helpers
from acme.utils import lp_utils
import launchpad as lp

FLAGS = flags.FLAGS
flags.DEFINE_string('task', 'MountainCarContinuous-v0',
                    'GYM environment task (str).')
flags.DEFINE_string(
    'dataset_name', 'd4rl_mujoco_halfcheetah/v0-medium', 'What dataset to use. '
    'See the TFDS catalog for possible values.')
flags.DEFINE_integer(
    'num_transitions_rb', 50000,
    'Number of demonstration transitions to put into the '
    'replay buffer.')
flags.DEFINE_integer('seed', 0, 'Random seed.')


def make_unbatched_demonstration_iterator(
    dataset_name: str) -> pwil.PWILDemonstrations:
  """Loads a demonstrations dataset and computes average episode length."""
  dataset = tfds.get_tfds_dataset(dataset_name)
  # Note: PWIL is not intended for large demonstration datasets.
  num_steps, num_episodes = functools.reduce(
      lambda accu, t: (accu[0] + 1, accu[1] + int(t.discount == 0.0)),
      dataset.as_numpy_iterator(), (0, 0))
  episode_length = num_steps / num_episodes if num_episodes else num_steps
  return pwil.PWILDemonstrations(dataset.as_numpy_iterator(), episode_length)


def main(_):
  task = FLAGS.task
  environment_factory = lambda seed: helpers.make_environment(task)
  sac_config = sac.SACConfig(num_sgd_steps_per_step=64)
  sac_builder = sac.SACBuilder(sac_config)
  pwil_config = pwil.PWILConfig(num_transitions_rb=FLAGS.num_transitions_rb)

  program = pwil.DistributedPWIL(
      environment_factory=environment_factory,
      rl_agent=sac_builder,
      config=pwil_config,
      network_factory=sac.make_networks,
      seed=FLAGS.seed,
      demonstrations_fn=functools.partial(
          make_unbatched_demonstration_iterator,
          dataset_name=FLAGS.dataset_name,
      ),
      policy_network=sac.apply_policy_and_sample,
      evaluator_policy_network=(
          lambda n: sac.apply_policy_and_sample(n, eval_mode=True)),
      num_actors=4,
      max_number_of_steps=100).build()

  # Launch experiment.
  lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))


if __name__ == '__main__':
  app.run(main)
