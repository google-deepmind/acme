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

"""Example running PWIL+SAC on the OpenAI Gym."""
import functools

from absl import flags
import acme
from acme import specs
from acme.agents.jax import pwil
from acme.agents.jax import sac
from acme.datasets import tfds
from absl import app
import helpers
from acme.utils import counting
from acme.utils import experiment_utils
import jax

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_steps', 1000000,
                     'Number of env steps to run training for.')
flags.DEFINE_integer('eval_every', 10000, 'How often to run evaluation')
flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'What environment to run')
flags.DEFINE_string(
    'dataset_name', 'd4rl_mujoco_halfcheetah/v0-medium', 'What dataset to use. '
    'See the TFDS catalog for possible values.')
flags.DEFINE_integer('num_sgd_steps_per_step', 1,
                     'Number of SGD steps per learner step() for SAC.')
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
  # Create an environment, grab the spec, and use it to create networks.
  environment = helpers.make_environment(task=FLAGS.env_name)
  environment_spec = specs.make_environment_spec(environment)

  # Construct the agent.
  sac_config = sac.SACConfig(
      target_entropy=sac.target_entropy_from_env_spec(environment_spec),
      num_sgd_steps_per_step=FLAGS.num_sgd_steps_per_step,
      min_replay_size=1,
      samples_per_insert_tolerance_rate=float('inf'))
  direct_rl_builder = sac.SACBuilder(sac_config)
  networks = sac.make_networks(environment_spec)
  policy_network = sac.apply_policy_and_sample(networks)
  evaluator_policy_network = (
      sac.apply_policy_and_sample(networks, eval_mode=True))
  batch_size = sac_config.batch_size * sac_config.num_sgd_steps_per_step

  agent = pwil.PWIL(
      spec=environment_spec,
      rl_agent=direct_rl_builder,
      config=pwil.PWILConfig(num_transitions_rb=FLAGS.num_transitions_rb),
      networks=networks,
      seed=FLAGS.seed,
      batch_size=batch_size,
      demonstrations_fn=functools.partial(
          make_unbatched_demonstration_iterator,
          dataset_name=FLAGS.dataset_name,
      ),
      policy_network=policy_network)

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
      policy_network=evaluator_policy_network,
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
