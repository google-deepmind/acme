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
from acme.agents.jax import dqn
from acme.agents.jax.dqn import losses
import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import launchpad as lp


RUN_DISTRIBUTED = flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
ENV_NAME = flags.DEFINE_string('env_name', 'Pong', 'What environment to run')
SEED = flags.DEFINE_integer('seed', 0, 'Random seed.')
NUM_STEPS = flags.DEFINE_integer('num_steps', 1_000_000,
                                 'Number of env steps to run.')


def build_experiment_config():
  """Builds MDQN experiment config which can be executed in different ways."""
  # Create an environment, grab the spec, and use it to create networks.
  env_name = ENV_NAME.value

  def env_factory(seed):
    del seed
    return helpers.make_atari_environment(
        level=env_name, sticky_actions=True, zero_discount_on_life_loss=False)

  # Construct the agent.
  config = dqn.DQNConfig(
      discount=0.99,
      eval_epsilon=0.,
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

  dqn_builder = dqn.DQNBuilder(config, loss_fn=loss_fn)

  return experiments.ExperimentConfig(
      builder=dqn_builder,
      environment_factory=env_factory,
      network_factory=helpers.make_dqn_atari_network,
      seed=SEED.value,
      max_num_actor_steps=NUM_STEPS.value)


def main(_):
  experiment_config = build_experiment_config()
  if RUN_DISTRIBUTED.value:
    program = experiments.make_distributed_experiment(
        experiment=experiment_config,
        num_actors=4 if lp_utils.is_local_run() else 128)
    lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
  else:
    experiments.run_experiment(experiment_config)


if __name__ == '__main__':
  app.run(main)
