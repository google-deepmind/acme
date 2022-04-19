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

"""Example running AIL+SAC in JAX on the OpenAI Gym.

It runs the distributed agent using Launchpad runtime specified by
--lp_launch_type flag.
"""

import functools

from absl import app
from absl import flags
from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import ail
from acme.agents.jax import sac
import helpers
from acme.jax import networks as networks_lib
from acme.utils import lp_utils
import haiku as hk
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

  ail_config = ail.AILConfig(direct_rl_batch_size=sac_config.batch_size *
                             sac_config.num_sgd_steps_per_step)

  def network_factory(spec: specs.EnvironmentSpec) -> ail.AILNetworks:

    def discriminator(*args, **kwargs) -> networks_lib.Logits:
      return ail.DiscriminatorModule(
          environment_spec=spec,
          use_action=True,
          use_next_obs=True,
          network_core=ail.DiscriminatorMLP([4, 4],))(*args, **kwargs)

    discriminator_transformed = hk.without_apply_rng(
        hk.transform_with_state(discriminator))

    return ail.AILNetworks(
        ail.make_discriminator(spec, discriminator_transformed),
        imitation_reward_fn=ail.rewards.gail_reward(),
        direct_rl_networks=sac.make_networks(spec))

  def policy_network(
      network: ail.AILNetworks,
      eval_mode: bool = False) -> actor_core_lib.FeedForwardPolicy:
    return sac.apply_policy_and_sample(
        network.direct_rl_networks, eval_mode=eval_mode)

  program = ail.DistributedAIL(
      environment_factory=environment_factory,
      rl_agent=sac_builder,
      config=ail_config,
      network_factory=network_factory,
      seed=0,
      batch_size=sac_config.batch_size * sac_config.num_sgd_steps_per_step,
      make_demonstrations=functools.partial(
          helpers.make_demonstration_iterator, dataset_name=FLAGS.dataset_name),
      policy_network=policy_network,
      evaluator_policy_network=(lambda n: policy_network(n, eval_mode=True)),
      num_actors=4,
      max_number_of_steps=100,
      discriminator_loss=ail.losses.gail_loss()).build()

  # Launch experiment.
  lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))


if __name__ == '__main__':
  app.run(main)
