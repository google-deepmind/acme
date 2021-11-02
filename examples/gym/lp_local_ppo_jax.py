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

"""Example running PPO in JAX on the OpenAI Gym.

It runs the distributed agent in the single-machine multi-process setup.
"""

from absl import app
from absl import flags
from acme.agents.jax import ppo
import helpers
import launchpad as lp

FLAGS = flags.FLAGS
flags.DEFINE_string('task', 'MountainCarContinuous-v0',
                    'GYM environment task (str).')
flags.DEFINE_integer('seed', 0, 'Random seed.')


def main(_):
  task = FLAGS.task
  environment_factory = lambda is_eval: helpers.make_environment(is_eval, task)
  config = ppo.PPOConfig(
      unroll_length=16,
      num_minibatches=32,
      num_epochs=10,
      batch_size=2048 // 16)
  program = ppo.DistributedPPO(
      environment_factory=environment_factory,
      network_factory=ppo.make_gym_networks,
      config=config,
      seed=FLAGS.seed,
      num_actors=4,
      max_number_of_steps=4000000).build()

  # Launch experiment.
  lp.launch(program, lp.LaunchType.LOCAL_MULTI_PROCESSING)


if __name__ == '__main__':
  app.run(main)
