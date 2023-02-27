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

"""Example running R2D2 on discrete control tasks."""

from absl import flags
from acme.agents.jax import r2d2
import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import dm_env
import launchpad as lp
from datetime import datetime, timedelta
from acme.jax import inference_server as inference_server_lib

# import ipdb; ipdb.set_trace()
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
  if physical_devices:
    for physical_device in physical_devices:
      tf.config.experimental.set_memory_growth(physical_device, True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


# Flags which modify the behavior of the launcher.
flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
flags.DEFINE_string('env_name', 'Pong', 'What environment to run.')
flags.DEFINE_integer('seed', 0, 'Random seed (experiment).')
flags.DEFINE_integer('num_actors', 64, 'Num actors if running distributed')
flags.DEFINE_integer('num_actors_per_node', 1, 'Actors per node (not sure what this means yet)')
flags.DEFINE_bool('multiprocessing_colocate_actors', False, 'Not sure, maybe whether to put actors in different processes?')
flags.DEFINE_integer('num_steps', 200_000_000,
                     'Number of environment steps to run for.')

FLAGS = flags.FLAGS


def build_experiment_config():
  """Builds R2D2 experiment config which can be executed in different ways."""
  # batch_size = 32
  batch_size = 8

  # The env_name must be dereferenced outside the environment factory as FLAGS
  # cannot be pickled and pickling is necessary when launching distributed
  # experiments via Launchpad.
  env_name = FLAGS.env_name

  # Create an environment factory.
  def environment_factory(seed: int) -> dm_env.Environment:
    del seed
    return helpers.make_atari_environment(
        level=env_name,
        sticky_actions=True,
        zero_discount_on_life_loss=False,
        oar_wrapper=True,
        num_stacked_frames=1,
        flatten_frame_stack=True,
        grayscaling=False)

  # Configure the agent.
  config = r2d2.R2D2Config(
      # burn_in_length=8,
      # trace_length=40,
      # sequence_period=20,
      burn_in_length=4,
      trace_length=20,
      sequence_period=10,
      # min_replay_size=10_000,
      min_replay_size=100,
      max_replay_size=10000,
      batch_size=batch_size,
      # prefetch_size=1,
      prefetch_size=0,
      # samples_per_insert=0,
      samples_per_insert=1,
      evaluation_epsilon=1e-3,
      # learning_rate=1e-4,
      # target_update_period=1200,
      # variable_update_period=100,
      # actor_jit=False,
      actor_jit=False,
  )

  return experiments.ExperimentConfig(
      builder=r2d2.R2D2Builder(config),
      network_factory=r2d2.make_atari_networks,
      environment_factory=environment_factory,
      seed=FLAGS.seed,
      max_num_actor_steps=FLAGS.num_steps)


def main(_):
  config = build_experiment_config()
  if FLAGS.run_distributed:
    num_actors = FLAGS.num_actors
    num_actors_per_node = FLAGS.num_actors_per_node
    inference_batch_size = int(max(num_actors_per_node//2, 1))
    print('inference batch size ', inference_batch_size)
    inference_server_config = inference_server_lib.InferenceServerConfig(
      # batch_size=max(num_actors_per_node // 2, 1),
      batch_size=inference_batch_size,
      update_period=400,
      timeout=timedelta(
          microseconds=999000,
      ),
      # timeout=1000,
      )
    print(inference_server_config)
    program = experiments.make_distributed_experiment(
        experiment=config, num_actors=num_actors,
        inference_server_config=inference_server_config,
        num_actors_per_node=num_actors_per_node, multiprocessing_colocate_actors=FLAGS.multiprocessing_colocate_actors)
    # program = experiments.make_distributed_experiment(
    #     experiment=config, num_actors=64 if lp_utils.is_local_run() else 80)
    lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program), terminal="current_terminal")
  else:
    experiments.run_experiment(experiment=config)


if __name__ == '__main__':
  start_time = datetime.now()
  app.run(main)
  end_time = datetime.now()
  print('End Time: {}'.format(end_time))
  print('Duration: {}'.format(end_time - start_time))

