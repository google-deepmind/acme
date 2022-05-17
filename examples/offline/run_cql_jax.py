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

"""An example CQL running on locomotion datasets (mujoco) from D4rl."""

from absl import app
from absl import flags
import acme
from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import cql
from acme.datasets import tfds
from acme.examples.offline import helpers as gym_helpers
from acme.jax import variable_utils
from acme.utils import loggers
import haiku as hk
import jax
import jax.numpy as jnp
import optax

# Agent flags
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('evaluate_every', 20, 'Evaluation period.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')
flags.DEFINE_integer(
    'num_demonstrations', 10,
    'Number of demonstration episodes to load from the dataset. If None, loads the full dataset.'
)
flags.DEFINE_integer('seed', 0, 'Random seed for learner and evaluator.')
# CQL specific flags.
flags.DEFINE_float('policy_learning_rate', 3e-5, 'Policy learning rate.')
flags.DEFINE_float('critic_learning_rate', 3e-4, 'Critic learning rate.')
flags.DEFINE_float('fixed_cql_coefficient', None,
                   'Fixed CQL coefficient. If None, an adaptive one is used.')
flags.DEFINE_float('cql_lagrange_threshold', 10.,
                   'Lagrange threshold for the adaptive CQL coefficient.')
# Environment flags.
flags.DEFINE_string('env_name', 'HalfCheetah-v2',
                    'Gym mujoco environment name.')
flags.DEFINE_string(
    'dataset_name', 'd4rl_mujoco_halfcheetah/v2-medium',
    'D4rl dataset name. Can be any locomotion dataset from '
    'https://www.tensorflow.org/datasets/catalog/overview#d4rl.')

FLAGS = flags.FLAGS


def main(_):
  key = jax.random.PRNGKey(FLAGS.seed)
  key_demonstrations, key_learner = jax.random.split(key, 2)

  # Create an environment and grab the spec.
  environment = gym_helpers.make_environment(task=FLAGS.env_name)
  environment_spec = specs.make_environment_spec(environment)

  # Get a demonstrations dataset.
  transitions_iterator = tfds.get_tfds_dataset(FLAGS.dataset_name,
                                               FLAGS.num_demonstrations)
  demonstrations = tfds.JaxInMemoryRandomSampleIterator(
      transitions_iterator, key=key_demonstrations, batch_size=FLAGS.batch_size)

  # Create the networks to optimize.
  networks = cql.make_networks(environment_spec)

  # Create the learner.
  learner = cql.CQLLearner(
      batch_size=FLAGS.batch_size,
      networks=networks,
      random_key=key_learner,
      policy_optimizer=optax.adam(FLAGS.policy_learning_rate),
      critic_optimizer=optax.adam(FLAGS.critic_learning_rate),
      fixed_cql_coefficient=FLAGS.fixed_cql_coefficient,
      cql_lagrange_threshold=FLAGS.cql_lagrange_threshold,
      demonstrations=demonstrations,
      num_sgd_steps_per_step=1)

  def evaluator_network(params: hk.Params, key: jnp.DeviceArray,
                        observation: jnp.DeviceArray) -> jnp.DeviceArray:
    dist_params = networks.policy_network.apply(params, observation)
    return networks.sample_eval(dist_params, key)

  actor_core = actor_core_lib.batched_feed_forward_to_actor_core(
      evaluator_network)
  variable_client = variable_utils.VariableClient(
      learner, 'policy', device='cpu')
  evaluator = actors.GenericActor(
      actor_core, key, variable_client, backend='cpu')

  eval_loop = acme.EnvironmentLoop(
      environment=environment,
      actor=evaluator,
      logger=loggers.TerminalLogger('evaluation', time_delta=0.))

  # Run the environment loop.
  while True:
    for _ in range(FLAGS.evaluate_every):
      learner.step()
    eval_loop.run(FLAGS.evaluation_episodes)


if __name__ == '__main__':
  app.run(main)
