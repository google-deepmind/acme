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
import functools

from absl import flags
import acme
from acme import specs
from acme.agents.jax import ail
from acme.agents.jax import ppo
from absl import app
import helpers
from acme.jax import networks as networks_lib
import haiku as hk
import jax

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_steps', 1000000,
                     'Number of env steps to run training for.')
flags.DEFINE_integer('transition_batch_size', 2048,
                     'Number of transitions in a batch.')
flags.DEFINE_integer('unroll_length', 16,
                     'Number of transitions per single gradient.')
flags.DEFINE_integer('eval_every', 10000, 'How often to run evaluation')
flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'What environment to run')
flags.DEFINE_string(
    'dataset_name', 'd4rl_mujoco_halfcheetah/v0-medium', 'What dataset to use. '
    'See the TFDS catalog for possible values.')
flags.DEFINE_integer('seed', 0, 'Random seed.')


def main(_):
  # Create an environment, grab the spec, and use it to create networks.
  environment = helpers.make_environment(task=FLAGS.env_name)
  environment_spec = specs.make_environment_spec(environment)
  agent_networks = ppo.make_gym_networks(environment_spec)

  # Construct the agent.
  ppo_config = ppo.PPOConfig(
      unroll_length=FLAGS.unroll_length,
      num_minibatches=32,
      num_epochs=10,
      batch_size=FLAGS.transition_batch_size // FLAGS.unroll_length)
  ppo_networks = ppo.make_gym_networks(environment_spec)

  ail_config = ail.AILConfig(
      direct_rl_batch_size=ppo_config.batch_size * ppo_config.unroll_length,
      discriminator_batch_size=FLAGS.transition_batch_size,
      is_sequence_based=True,
      num_sgd_steps_per_step=1,
      share_iterator=True,
  )

  def discriminator(*args, **kwargs) -> networks_lib.Logits:
    # Note: observation embedding does not seem needed.
    # embedding = lambda x: jnp.reshape(x, list(x.shape[:-2]) + [-1])
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
  train_loop = acme.EnvironmentLoop(environment, agent, label='train_loop')
  # Create the evaluation actor and loop.
  eval_actor = agent.builder.make_actor(
      random_key=jax.random.PRNGKey(FLAGS.seed),
      policy_network=ppo.make_inference_fn(agent_networks, evaluation=True),
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
