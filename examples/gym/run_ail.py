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

"""Example running AIL+SAC on the OpenAI Gym."""
import dataclasses
import functools

from absl import flags
import acme
from acme import specs
from acme.agents.jax import ail
from acme.agents.jax import bc
from acme.agents.jax import sac
from absl import app
import helpers
from acme.jax import networks as networks_lib
import haiku as hk
import jax

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_steps', 1000000,
                     'Number of env steps to run training for.')
flags.DEFINE_integer('eval_every', 10000, 'How often to run evaluation')
flags.DEFINE_string('env_name', 'MountainCarContinuous-v0',
                    'What environment to run')
flags.DEFINE_string(
    'dataset_name', 'd4rl_mujoco_halfcheetah/v0-medium', 'What dataset to use. '
    'See the TFDS catalog for possible values.')
flags.DEFINE_integer('num_sgd_steps_per_step', 1,
                     'Number of SGD steps per learner step().')
flags.DEFINE_integer('seed', 0, 'Random seed.')


def add_bc_pretraining(sac_networks: sac.SACNetworks) -> sac.SACNetworks:
  """Augments `sac_networks` to run BC pretraining in policy_network.init."""

  make_demonstrations = functools.partial(
      helpers.make_demonstration_iterator, dataset_name=FLAGS.dataset_name)
  bc_network = bc.pretraining.convert_to_bc_network(sac_networks.policy_network)
  loss = bc.logp(sac_networks.log_prob)

  def bc_init(*unused_args):
    return bc.pretraining.train_with_bc(make_demonstrations, bc_network, loss)

  return dataclasses.replace(
      sac_networks,
      policy_network=networks_lib.FeedForwardNetwork(
          bc_init, sac_networks.policy_network.apply))


def main(_):
  # Create an environment, grab the spec, and use it to create networks.
  environment = helpers.make_environment(task=FLAGS.env_name)
  environment_spec = specs.make_environment_spec(environment)
  agent_networks = sac.make_networks(environment_spec)

  # Construct the agent.
  # Local layout makes sure that we populate the buffer with min_replay_size
  # initial transitions and that there's no need for tolerance_rate. In order
  # for deadlocks not to happen we need to disable rate limiting that heppens
  # inside the SACBuilder. This is achieved by the min_replay_size and
  # samples_per_insert_tolerance_rate arguments.
  sac_config = sac.SACConfig(
      target_entropy=sac.target_entropy_from_env_spec(environment_spec),
      num_sgd_steps_per_step=FLAGS.num_sgd_steps_per_step,
      min_replay_size=1,
      samples_per_insert_tolerance_rate=float('inf'))
  sac_builder = sac.SACBuilder(sac_config)
  sac_networks = sac.make_networks(environment_spec)
  sac_networks = add_bc_pretraining(sac_networks)

  ail_config = ail.AILConfig(direct_rl_batch_size=sac_config.batch_size *
                             sac_config.num_sgd_steps_per_step)

  def discriminator(*args, **kwargs) -> networks_lib.Logits:
    return ail.DiscriminatorModule(
        environment_spec=environment_spec,
        use_action=True,
        use_next_obs=True,
        network_core=ail.DiscriminatorMLP([4, 4],))(*args, **kwargs)

  discriminator_transformed = hk.without_apply_rng(
      hk.transform_with_state(discriminator))

  ail_network = ail.AILNetworks(
      ail.make_discriminator(environment_spec, discriminator_transformed),
      imitation_reward_fn=ail.rewards.gail_reward(),
      direct_rl_networks=sac_networks)

  agent = ail.AIL(
      spec=environment_spec,
      rl_agent=sac_builder,
      network=ail_network,
      config=ail_config,
      seed=FLAGS.seed,
      batch_size=sac_config.batch_size * sac_config.num_sgd_steps_per_step,
      make_demonstrations=functools.partial(
          helpers.make_demonstration_iterator, dataset_name=FLAGS.dataset_name),
      policy_network=sac.apply_policy_and_sample(sac_networks),
      discriminator_loss=ail.losses.gail_loss())

  # Create the environment loop used for training.
  train_loop = acme.EnvironmentLoop(environment, agent, label='train_loop')
  # Create the evaluation actor and loop.
  eval_actor = agent.builder.make_actor(
      random_key=jax.random.PRNGKey(FLAGS.seed),
      policy_network=sac.apply_policy_and_sample(
          agent_networks, eval_mode=True),
      variable_source=agent)
  eval_env = helpers.make_environment(task=FLAGS.env_name)
  eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label='eval_loop')

  assert FLAGS.num_steps % FLAGS.eval_every == 0
  for _ in range(FLAGS.num_steps // FLAGS.eval_every):
    eval_loop.run(num_episodes=5)
    train_loop.run(num_steps=FLAGS.eval_every)
  eval_loop.run(num_episodes=5)


if __name__ == '__main__':
  app.run(main)
