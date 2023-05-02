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

import os
import signal
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
from get_local_resources import _get_local_resources
start_time = datetime.now()

# Flags which modify the behavior of the launcher.
flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
flags.DEFINE_string('env_name', 'Pong', 'What environment to run.')
flags.DEFINE_integer('seed', 0, 'Random seed (experiment).')
flags.DEFINE_integer('num_actors', 64, 'Num actors if running distributed')
flags.DEFINE_integer('inference_batch_size', 0, 'Inference batch size')
flags.DEFINE_boolean('use_inference_server', False, 'Whether we use inference server (default False, include with no args to be true)')
flags.DEFINE_integer('num_inference_servers', 1, 'Number of inference servers (defaults to 1)')
flags.DEFINE_integer('num_actors_per_node', 1, 'Actors per node (not sure what this means yet)')
flags.DEFINE_boolean('multiprocessing_colocate_actors', False, 'Not sure, maybe whether to put actors in different processes?')
flags.DEFINE_boolean('actors_on_gpu', False, 'Whether we put actors on GPU (default is on CPU)')
flags.DEFINE_boolean('learner_on_cpu', False, 'For testing whether learner on GPU makes inference faster')
flags.DEFINE_integer('num_steps', 50_000_000,
                     'Number of environment steps to run for.')
flags.DEFINE_float('spi', 1.0,
                     'Number of samples per insert. 0 means does not constrain, other values do.')

flags.DEFINE_integer('actor_cpu_start', -1, "If we're partitioning actors by CPU")
flags.DEFINE_integer('actor_cpu_end', -1, "If we're partitioning actors by CPU")
flags.DEFINE_string('acme_id', None, 'Experiment identifier to use for Acme.')

FLAGS = flags.FLAGS


def build_experiment_config():
  """Builds R2D2 experiment config which can be executed in different ways."""
  batch_size = 32
  # batch_size = 8

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

  # Their default config:
  config = r2d2.R2D2Config(
      burn_in_length=8,
      trace_length=40,
      sequence_period=20,
      # min_replay_size=10_000,
      min_replay_size=1000,
      batch_size=batch_size,
      prefetch_size=1,
      # samples_per_insert=1.0,
      samples_per_insert= FLAGS.spi,
      evaluation_epsilon=1e-3,
      learning_rate=1e-4,
      target_update_period=1200,
      variable_update_period=100,
      actor_jit=not FLAGS.use_inference_server, # we don't use this if we're doing inference-server
  )

  # # Configure the agent.

  # batch_size = 8
  # config = r2d2.R2D2Config(
  #     # burn_in_length=8,
  #     # trace_length=40,
  #     # sequence_period=20,
  #     burn_in_length=2,
  #     trace_length=10,
  #     sequence_period=4,
  #     # min_replay_size=10_000,
  #     # min_replay_size=200,
  #     # max_replay_size=10000,
  #     min_replay_size=10000,
  #     max_replay_size=100000,
  #     batch_size=batch_size,
  #     # prefetch_size=1,
  #     prefetch_size=0,
  #     samples_per_insert=0,
  #     # samples_per_insert=1,
  #     # samples_per_insert=4.0, # shouldn't this be 0.25 to match DQN? I dunno. Maybe this is more sample efficent.
  #     # can see what it means/does.
  #     evaluation_epsilon=1e-3,
  #     # learning_rate=1e-4,
  #     # target_update_period=1200,
  #     # variable_update_period=100,
  #     # actor_jit=False,
  #     # actor_jit=False,
  #     actor_jit=not FLAGS.use_inference_server, # we don't use this if we're doing inference-server
  # )

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
       "actor":PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(0)}),
       "evaluator":PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(0)}),
       "inference_server":PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)}),
       "counter":PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)}),
       "replay":PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)}),
     }
   else:
     local_resources = {}
   return local_resources


def sigterm_log_endtime_handler(_signo, _stack_frame):
  """
  log end time gracefully on SIGTERM
  we use SIGTERM because this is what acme throws when the experiments end by reaching nun_steps
  """
  end_time = datetime.now()
  # log start and end time 
  log_dir = os.path.expanduser(os.path.join('~/acme', FLAGS.acme_id))
  # don't print because it will be lost, especially because acme stops experiment by throwing an Error when reaching num_steps
  from helpers import save_start_and_end_time
  save_start_and_end_time(log_dir, start_time, end_time)

  # log the command used
  from helpers import save_command_used
  save_command_used(log_dir)

  # log git stuff
  from helpers import is_under_git_control, save_git_information
  if is_under_git_control():
      save_git_information(log_dir)


def main(_):
  import os
  FLAGS.append_flags_into_file('/tmp/temp_flags')  # hack: so that subprocesses can load FLAGS
  config = build_experiment_config()
  print(FLAGS.use_inference_server)
  if FLAGS.run_distributed:
    num_actors = FLAGS.num_actors
    num_actors_per_node = FLAGS.num_actors_per_node
    inference_batch_size = FLAGS.inference_batch_size or int(max(num_actors//2, 1)) # defaults flag to 0
    launch_type = FLAGS.lp_launch_type
    if FLAGS.use_inference_server:
      print('inference batch size ', inference_batch_size)
      # print('but not using it')
      inference_server_config = inference_server_lib.InferenceServerConfig(
        # batch_size=max(num_actors_per_node // 2, 1),
        batch_size=inference_batch_size, 
        update_period=400,
        # update_period=5,
        timeout=timedelta(
            microseconds=999000,
        ),
        # timeout=1000,
        )
      print(inference_server_config)
    else:
      inference_server_config = None
      print('not using inference server')
    local_resources = _get_local_resources(launch_type)
    print(local_resources)
    program = experiments.make_distributed_experiment(
        experiment=config, num_actors=num_actors,
        inference_server_config=inference_server_config,
        num_inference_servers=FLAGS.num_inference_servers,
        num_actors_per_node=num_actors_per_node,
        multiprocessing_colocate_actors=FLAGS.multiprocessing_colocate_actors,
        split_actor_cpus=(FLAGS.actor_cpu_start >= 0 and FLAGS.actor_cpu_end >= 0)
        )
    # program = experiments.make_distributed_experiment(
    #     experiment=config, num_actors=64 if lp_utils.is_local_run() else 80)
    lp.launch(program,
              xm_resources=lp_utils.make_xm_docker_resources(program),
              local_resources=local_resources,
              # terminal="current_terminal")
              terminal="tmux_session")
  else:
    experiments.run_experiment(experiment=config)
  

if __name__ == '__main__':
  signal.signal(signal.SIGTERM, sigterm_log_endtime_handler)
  app.run(main)

