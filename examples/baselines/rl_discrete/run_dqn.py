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

"""Example running DQN on discrete control tasks."""

from absl import flags
from acme.agents.jax import dqn
from acme.agents.jax.dqn import losses
import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import launchpad as lp
from get_local_resources import _get_local_resources
from acme.utils.experiment_utils import make_experiment_logger
import functools

RUN_DISTRIBUTED = flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
ENV_NAME = flags.DEFINE_string('env_name', 'Pong', 'What environment to run')
SEED = flags.DEFINE_integer('seed', 0, 'Random seed.')
NUM_STEPS = flags.DEFINE_integer('num_steps', 1_000_000,
                                 'Number of env steps to run.')
SPI = flags.DEFINE_float('spi', 8,
                         'Number of samples per insert.')
NUM_ACTORS = flags.DEFINE_integer('num_actors', 64,
                                 'Number of actors to use.')
# Needed inside of get_local_resources.
flags.DEFINE_integer('num_actors_per_node', 1,
                      'Number of actors to use.')
flags.DEFINE_boolean('one_cpu_per_actor', False, 'If we pin each actor to a different CPU')
flags.DEFINE_list("actor_gpu_ids", ["-1"], "Which GPUs to use for actors. Actors select GPU in round-robin fashion")
flags.DEFINE_list("learner_gpu_ids", ["0"], "Which GPUs to use for learner. Gets all")
flags.DEFINE_string('acme_id', None, 'Experiment identifier to use for Acme.')
flags.DEFINE_string('acme_dir', '~/acme', 'Directory to do acme logging')
flags.DEFINE_boolean('use_inference_server', False, 'Whether we use inference server (default False, include with no args to be true)')
flags.DEFINE_list("inference_server_gpu_ids", ["1"], "Which GPUs to use for inference servers. For now, all get all")

FLAGS = flags.FLAGS

def build_experiment_config():
  """Builds DQN experiment config which can be executed in different ways."""
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
      # samples_per_insert=8,
      samples_per_insert=SPI.value,
      batch_size=32)
  loss_fn = losses.QLearning(
      discount=config.discount, max_abs_reward=1.)

  dqn_builder = dqn.DQNBuilder(config, loss_fn=loss_fn)
  checkpointing_config = experiments.CheckpointingConfig(directory=FLAGS.acme_dir)

  return experiments.ExperimentConfig(
      builder=dqn_builder,
      environment_factory=env_factory,
      network_factory=helpers.make_dqn_atari_network,
      seed=SEED.value,
      max_num_actor_steps=NUM_STEPS.value,
      checkpointing=checkpointing_config,
      logger_factory=functools.partial(make_experiment_logger, save_dir=FLAGS.acme_dir))


def main(_):
  launch_type = FLAGS.lp_launch_type
  local_resources = _get_local_resources(launch_type)
  experiment_config = build_experiment_config()
  if RUN_DISTRIBUTED.value:
    program = experiments.make_distributed_experiment(
      experiment=experiment_config,
      num_actors=NUM_ACTORS.value,
      split_actor_specs=True,
    )
    lp.launch(
      program, xm_resources=lp_utils.make_xm_docker_resources(program),
      local_resources=local_resources,
      terminal="tmux_session")
  else:
    raise Exception("We don't do this here")
    experiments.run_experiment(experiment_config)


if __name__ == '__main__':
  app.run(main)
