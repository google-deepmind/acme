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

"""Example running MuZero on discrete control tasks."""

import datetime
import math

from absl import flags
from acme import specs
from acme.agents.jax import muzero
import helpers
from absl import app
from acme.jax import experiments
from acme.jax import inference_server as inference_server_lib
from acme.utils import lp_utils
import dm_env
import launchpad as lp


ENV_NAME = flags.DEFINE_string('env_name', 'Pong', 'What environment to run')
SEED = flags.DEFINE_integer('seed', 0, 'Random seed.')
NUM_STEPS = flags.DEFINE_integer(
    'num_steps', 2_000_000, 'Number of env steps to run.'
)
NUM_LEARNERS = flags.DEFINE_integer('num_learners', 1, 'Number of learners.')
NUM_ACTORS = flags.DEFINE_integer('num_actors', 4, 'Number of actors.')
NUM_ACTORS_PER_NODE = flags.DEFINE_integer(
    'num_actors_per_node',
    2,
    'Number of colocated actors',
)
RUN_DISTRIBUTED = flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.',)


def build_experiment_config() -> experiments.ExperimentConfig:
  """Builds DQN experiment config which can be executed in different ways."""
  env_name = ENV_NAME.value
  muzero_config = muzero.MZConfig()

  def env_factory(seed: int) -> dm_env.Environment:
    del seed
    return helpers.make_atari_environment(
        level=env_name,
        sticky_actions=True,
        zero_discount_on_life_loss=True,
        num_stacked_frames=1,
        grayscaling=False,
        to_float=False,
    )

  def network_factory(
      spec: specs.EnvironmentSpec,
  ) -> muzero.MzNetworks:
    return muzero.make_network(
        spec,
        stack_size=muzero_config.stack_size,
    )

  # Construct the builder.
  env_spec = specs.make_environment_spec(env_factory(SEED.value))
  extra_spec = {
      muzero.POLICY_PROBS_KEY: specs.Array(
          shape=(env_spec.actions.num_values,), dtype='float32'
      ),
      muzero.RAW_VALUES_KEY: specs.Array(shape=(), dtype='float32'),
  }
  muzero_builder = muzero.MzBuilder(  # pytype: disable=wrong-arg-types  # jax-ndarray
      muzero_config,
      extra_spec,
  )

  checkpointing_config = experiments.CheckpointingConfig(
      replay_checkpointing_time_delta_minutes=20,
      time_delta_minutes=1,
  )
  return experiments.ExperimentConfig(
      builder=muzero_builder,
      environment_factory=env_factory,
      network_factory=network_factory,
      seed=SEED.value,
      max_num_actor_steps=NUM_STEPS.value,
      checkpointing=checkpointing_config,
  )


def main(_):
  experiment_config = build_experiment_config()

  if not RUN_DISTRIBUTED.value:
    raise NotImplementedError('Single threaded experiment not supported.')

  inference_server_config = inference_server_lib.InferenceServerConfig(
      batch_size=64,
      update_period=400,
      timeout=datetime.timedelta(
          seconds=1,
      ),
  )
  num_inference_servers = math.ceil(
      NUM_ACTORS.value / (128 * NUM_ACTORS_PER_NODE.value),
  )

  program = experiments.make_distributed_experiment(
      experiment=experiment_config,
      num_actors=NUM_ACTORS.value,
      num_learner_nodes=NUM_LEARNERS.value,
      num_actors_per_node=NUM_ACTORS_PER_NODE.value,
      num_inference_servers=num_inference_servers,
      inference_server_config=inference_server_config,
  )
  lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program,),)


if __name__ == '__main__':
  app.run(main)
