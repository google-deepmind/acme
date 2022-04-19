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

"""Example running RND+SAC in JAX on the OpenAI Gym.

It runs the distributed agent using Launchpad runtime specified by
--lp_launch_type flag.
"""

from absl import app
from absl import flags
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import rnd
from acme.agents.jax import sac
import helpers
from acme.utils import lp_utils
import launchpad as lp

FLAGS = flags.FLAGS
flags.DEFINE_string('task', 'MountainCarContinuous-v0',
                    'GYM environment task (str).')


def main(_):
  task = FLAGS.task
  environment_factory = lambda seed: helpers.make_environment(task)
  num_sgd_steps_per_step = 64
  sac_config = sac.SACConfig(num_sgd_steps_per_step=num_sgd_steps_per_step)
  sac_builder = sac.SACBuilder(sac_config)
  rnd_config = rnd.RNDConfig(num_sgd_steps_per_step=num_sgd_steps_per_step)
  network_factory = lambda spec: rnd.make_networks(spec, sac.make_networks(spec)
                                                  )

  def policy_network(
      network: rnd.RNDNetworks,
      eval_mode: bool = False) -> actor_core_lib.FeedForwardPolicy:
    return sac.apply_policy_and_sample(
        network.direct_rl_networks, eval_mode=eval_mode)

  program = rnd.DistributedRND(
      environment_factory=environment_factory,
      rl_agent=sac_builder,
      network_factory=network_factory,
      config=rnd_config,
      policy_network=policy_network,
      evaluator_policy_network=(lambda n: policy_network(n, eval_mode=True)),
      num_actors=4,
      max_number_of_steps=1000000).build()

  # Launch experiment.
  lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))


if __name__ == '__main__':
  app.run(main)
