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

"""Example running Munchausen-DQN on discrete control tasks."""

from absl import flags
import acme
from acme import specs
from acme.agents.jax import dqn
from acme.agents.jax.dqn import agents as dqn_agent
from acme.agents.jax.dqn import losses
import helpers
from absl import app
from acme.utils import counting
from acme.utils import experiment_utils
import atari_py  # pylint:disable=unused-import

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'Pong', 'What environment to run')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_steps', 1_000_000, 'Number of env steps to run.')


def main(_):
  # Create an environment, grab the spec, and use it to create networks.
  environment = helpers.make_atari_environment(
      level=FLAGS.env_name,
      sticky_actions=True,
      zero_discount_on_life_loss=False)
  environment_spec = specs.make_environment_spec(environment)

  # Create network.
  network = helpers.make_dqn_atari_network(environment_spec)

  # Construct the agent.
  config = dqn.DQNConfig(
      discount=0.99,
      learning_rate=5e-5,
      n_step=1,
      epsilon=0.01,
      target_update_period=2000,
      min_replay_size=20_000,
      max_replay_size=1_000_000,
      samples_per_insert=8,
      batch_size=32)
  loss_fn = losses.MunchausenQLearning(
      discount=config.discount, max_abs_reward=1., huber_loss_parameter=1.,
      entropy_temperature=0.03, munchausen_coefficient=0.9)
  agent = dqn_agent.DQN(
      environment_spec,
      network,
      loss_fn=loss_fn,
      config=config,
      seed=FLAGS.seed)

  # Create the environment loop used for training.
  logger = experiment_utils.make_experiment_logger(
      label='train', steps_key='train_steps')
  train_loop = acme.EnvironmentLoop(
      environment,
      agent,
      label='train',
      counter=counting.Counter(prefix='train'),
      logger=logger)

  train_loop.run(num_steps=FLAGS.num_steps)


if __name__ == '__main__':
  app.run(main)
