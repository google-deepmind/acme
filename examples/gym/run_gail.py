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

"""Example running AIL+PPO on the OpenAI Gym."""
import dataclasses
import functools

from absl import flags
import acme
from acme import specs
from acme.agents.jax import ail
from acme.agents.jax import bc
from acme.agents.jax import ppo
from absl import app
import helpers
from acme.jax import networks as networks_lib
from acme.utils import counting
from acme.utils import experiment_utils
import haiku as hk
import jax

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_steps', 1000000,
                     'Number of env steps to run training for.')
flags.DEFINE_integer('num_discriminator_steps_per_step', 32,
                     'Number of discriminator training steps to balance it '
                     'against generation.')
flags.DEFINE_integer('ppo_num_minibatches', 32,
                     'Number of PPO minibatches per batch.')
flags.DEFINE_integer('ppo_num_epochs', 10,
                     'Number of PPO batch epochs per step.')
flags.DEFINE_integer('transition_batch_size', 512,
                     'Number of transitions in a batch.')
flags.DEFINE_integer('unroll_length', 2,
                     'Number of transitions per single gradient.')
flags.DEFINE_integer('eval_every', 40000, 'How often to run evaluation')
flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'What environment to run')
flags.DEFINE_string(
    'dataset_name', 'd4rl_mujoco_halfcheetah/v0-medium', 'What dataset to use. '
    'See the TFDS catalog for possible values.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_bool('pretrain', True, 'Whether to do BC pre-training.')
flags.DEFINE_bool('share_iterator', True,
                  'Whether to use a single Reverb iterator for the '
                  'discirminator and the RL policy learner.')


def add_bc_pretraining(ppo_networks: ppo.PPONetworks) -> ppo.PPONetworks:
  """Augments `ppo_networks` to run BC pretraining in policy_network.init."""

  make_demonstrations = functools.partial(
      helpers.make_demonstration_iterator, dataset_name=FLAGS.dataset_name)
  bc_network = bc.pretraining.convert_policy_value_to_bc_network(
      ppo_networks.network)
  loss = bc.logp(ppo_networks.log_prob)

  # Note: despite only training the policy network, this will also include the
  # initial value network params.
  def bc_init(*unused_args):
    return bc.pretraining.train_with_bc(make_demonstrations, bc_network, loss)

  return dataclasses.replace(
      ppo_networks,
      network=networks_lib.FeedForwardNetwork(
          bc_init, ppo_networks.network.apply))


def main(_):
  # Create an environment, grab the spec, and use it to create networks.
  environment = helpers.make_environment(task=FLAGS.env_name)
  environment_spec = specs.make_environment_spec(environment)
  agent_networks = ppo.make_gym_networks(environment_spec)

  # Construct the agent.
  ppo_config = ppo.PPOConfig(
      unroll_length=FLAGS.unroll_length,
      num_minibatches=FLAGS.ppo_num_minibatches,
      num_epochs=FLAGS.ppo_num_epochs,
      batch_size=FLAGS.transition_batch_size // FLAGS.unroll_length,
      learning_rate=0.0003,
      entropy_cost=0,
      gae_lambda=0.8,
      value_cost=0.25)
  ppo_networks = ppo.make_gym_networks(environment_spec)
  if FLAGS.pretrain:
    ppo_networks = add_bc_pretraining(ppo_networks)

  discriminator_batch_size = FLAGS.transition_batch_size
  ail_config = ail.AILConfig(
      direct_rl_batch_size=ppo_config.batch_size * ppo_config.unroll_length,
      discriminator_batch_size=discriminator_batch_size,
      is_sequence_based=True,
      num_sgd_steps_per_step=FLAGS.num_discriminator_steps_per_step,
      share_iterator=FLAGS.share_iterator,
  )

  def discriminator(*args, **kwargs) -> networks_lib.Logits:
    # Note: observation embedding is not needed for e.g. Mujoco.
    return ail.DiscriminatorModule(
        environment_spec=environment_spec,
        use_action=True,
        use_next_obs=True,
        network_core=ail.DiscriminatorMLP([4, 4],),
    )(*args, **kwargs)

  discriminator_transformed = hk.without_apply_rng(
      hk.transform_with_state(discriminator))

  ail_network = ail.AILNetworks(
      ail.make_discriminator(environment_spec, discriminator_transformed),
      imitation_reward_fn=ail.rewards.gail_reward(),
      direct_rl_networks=ppo_networks)

  agent = ail.GAIL(
      spec=environment_spec,
      network=ail_network,
      config=ail.GAILConfig(ail_config, ppo_config),
      seed=FLAGS.seed,
      batch_size=ppo_config.batch_size,
      make_demonstrations=functools.partial(
          helpers.make_demonstration_iterator, dataset_name=FLAGS.dataset_name),
      policy_network=ppo.make_inference_fn(ppo_networks))

  # Create the environment loop used for training.
  train_logger = experiment_utils.make_experiment_logger(
      label='train', steps_key='train_steps')
  train_loop = acme.EnvironmentLoop(
      environment,
      agent,
      counter=counting.Counter(prefix='train'),
      logger=train_logger)

  # Create the evaluation actor and loop.
  eval_logger = experiment_utils.make_experiment_logger(
      label='eval', steps_key='eval_steps')
  eval_actor = agent.builder.make_actor(
      random_key=jax.random.PRNGKey(FLAGS.seed),
      policy_network=ppo.make_inference_fn(agent_networks, evaluation=True),
      variable_source=agent)
  eval_env = helpers.make_environment(task=FLAGS.env_name)
  eval_loop = acme.EnvironmentLoop(
      eval_env,
      eval_actor,
      counter=counting.Counter(prefix='eval'),
      logger=eval_logger)

  assert FLAGS.num_steps % FLAGS.eval_every == 0
  for _ in range(FLAGS.num_steps // FLAGS.eval_every):
    eval_loop.run(num_episodes=5)
    train_loop.run(num_steps=FLAGS.eval_every)
  eval_loop.run(num_episodes=5)


if __name__ == '__main__':
  app.run(main)
