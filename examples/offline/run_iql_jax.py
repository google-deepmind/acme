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

"""Example IQL agent running on D4RL locomotion datasets."""

from absl import app
from absl import flags
import acme
from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import iql
from acme.datasets import tfds
from acme.examples.offline import helpers as gym_helpers
from acme.jax import variable_utils
from acme.utils import loggers
import haiku as hk
import jax
import optax

# Agent flags
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('evaluate_every', 20, 'Evaluation period.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')
flags.DEFINE_integer(
    'num_demonstrations', None,
    'Number of demonstration episodes to load. If None, loads full dataset.')
flags.DEFINE_integer('seed', 0, 'Random seed for learner and evaluator.')

# IQL specific flags
flags.DEFINE_float('policy_learning_rate', 3e-4, 'Policy learning rate.')
flags.DEFINE_float('value_learning_rate', 3e-4, 'Value function learning rate.')
flags.DEFINE_float('critic_learning_rate', 3e-4, 'Q-function learning rate.')
flags.DEFINE_float('expectile', 0.7,
                   'Expectile for value function. Higher is more conservative.')
flags.DEFINE_float('temperature', 3.0,
                   'Temperature for advantage weighting. Higher gives more weight to high advantages.')
flags.DEFINE_float('tau', 0.005, 'Target network update coefficient.')
flags.DEFINE_float('discount', 0.99, 'Discount factor.')

# Environment flags
flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Gym mujoco environment name.')
flags.DEFINE_string(
    'dataset_name', 'd4rl_mujoco_halfcheetah/v2-medium',
    'D4RL dataset name. Can be any locomotion dataset from '
    'https://www.tensorflow.org/datasets/catalog/overview#d4rl.')

FLAGS = flags.FLAGS


def main(_):
  key = jax.random.PRNGKey(FLAGS.seed)
  key_demonstrations, key_learner = jax.random.split(key, 2)

  # Create environment and get specification
  environment = gym_helpers.make_environment(task=FLAGS.env_name)
  environment_spec = specs.make_environment_spec(environment)

  # Load demonstrations dataset
  transitions_iterator = tfds.get_tfds_dataset(
      FLAGS.dataset_name, FLAGS.num_demonstrations)
  demonstrations = tfds.JaxInMemoryRandomSampleIterator(
      transitions_iterator,
      key=key_demonstrations,
      batch_size=FLAGS.batch_size)

  # Create networks
  networks = iql.make_networks(environment_spec)

  # Create learner
  learner = iql.IQLLearner(
      batch_size=FLAGS.batch_size,
      networks=networks,
      random_key=key_learner,
      demonstrations=demonstrations,
      policy_optimizer=optax.adam(FLAGS.policy_learning_rate),
      value_optimizer=optax.adam(FLAGS.value_learning_rate),
      critic_optimizer=optax.adam(FLAGS.critic_learning_rate),
      tau=FLAGS.tau,
      expectile=FLAGS.expectile,
      temperature=FLAGS.temperature,
      discount=FLAGS.discount,
      num_sgd_steps_per_step=1)

  def evaluator_network(
      params: hk.Params,
      key: jax.Array,
      observation: jax.Array) -> jax.Array:
    """Evaluation policy (deterministic)."""
    dist_params = networks.policy_network.apply(params, observation)
    return networks.sample_eval(dist_params, key)

  # Create evaluator actor
  actor_core = actor_core_lib.batched_feed_forward_to_actor_core(
      evaluator_network)
  variable_client = variable_utils.VariableClient(
      learner, 'policy', device='cpu')
  evaluator = actors.GenericActor(
      actor_core, key, variable_client, backend='cpu')

  # Create evaluation loop
  eval_loop = acme.EnvironmentLoop(
      environment=environment,
      actor=evaluator,
      logger=loggers.TerminalLogger('evaluation', time_delta=0.))

  # Training loop
  print(f'Training IQL on {FLAGS.dataset_name}...')
  print(f'Hyperparameters: expectile={FLAGS.expectile}, temperature={FLAGS.temperature}')
  
  while True:
    for _ in range(FLAGS.evaluate_every):
      learner.step()
    eval_loop.run(FLAGS.evaluation_episodes)


if __name__ == '__main__':
  app.run(main)
