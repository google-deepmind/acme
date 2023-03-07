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
from datetime import datetime

# Flags which modify the behavior of the launcher.
flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
flags.DEFINE_string('env_name', 'Pong', 'What environment to run.')
flags.DEFINE_integer('seed', 0, 'Random seed (experiment).')
flags.DEFINE_integer('num_steps', 200_000_000,
                     'Number of environment steps to run for.')
flags.DEFINE_integer('num_actors', 64, 'Number of actors to use')

FLAGS = flags.FLAGS


def build_experiment_config():
  """Builds R2D2 experiment config which can be executed in different ways."""
  batch_size = 32

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
      burn_in_length=8,
      trace_length=40,
      sequence_period=20,
      min_replay_size=10_000,
      batch_size=batch_size,
      prefetch_size=1,
      samples_per_insert=0,
      evaluation_epsilon=1e-3,
      learning_rate=1e-4,
      target_update_period=1200,
      variable_update_period=100,
  )

  return experiments.ExperimentConfig(
      builder=r2d2.R2D2Builder(config),
      network_factory=r2d2.make_atari_networks,
      environment_factory=environment_factory,
      seed=FLAGS.seed,
      max_num_actor_steps=FLAGS.num_steps)


def _get_local_resources(launch_type):
   assert launch_type in ('local_mp', 'local_mt'), launch_type
   from launchpad.nodes.python.local_multi_processing import PythonProcess
   if launch_type == 'local_mp':
     local_resources = {
       "learner":PythonProcess(env={
         "CUDA_VISIBLE_DEVICES": str(0),
         "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
         "TF_FORCE_GPU_ALLOW_GROWTH": "true",
       }),
       "actor":PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)}),
       "evaluator":PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(0)}),
       "inference_server":PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)}),
       "counter":PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)}),
       "replay":PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)}),
     }
   else:
     local_resources = {}
   return local_resources


def main(_):
  config = build_experiment_config()
  if FLAGS.run_distributed:
    program = experiments.make_distributed_experiment(
        experiment=config, num_actors=FLAGS.num_actors if lp_utils.is_local_run() else 80
    )
    lp.launch(program, 
              xm_resources=lp_utils.make_xm_docker_resources(program),
              local_resources=_get_local_resources(FLAGS.lp_launch_type),
              terminal='current_terminal')
  else:
    experiments.run_experiment(experiment=config)


if __name__ == '__main__':
  start_time = datetime.now()
  app.run(main)
  end_time = datetime.now()
  print('End Time: {}'.format(end_time))
  print('Duration: {}'.format(end_time - start_time))

