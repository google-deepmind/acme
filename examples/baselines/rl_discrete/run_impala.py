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

"""Example running IMPALA on discrete control tasks."""

from absl import flags
from acme.agents.jax import impala
from acme.agents.jax.impala import builder as impala_builder
import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import launchpad as lp
import optax


# Flags which modify the behavior of the launcher.
RUN_DISTRIBUTED = flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
ENV_NAME = flags.DEFINE_string('env_name', 'Pong', 'What environment to run.')
SEED = flags.DEFINE_integer('seed', 0, 'Random seed (experiment).')
NUM_ACTOR_STEPS = flags.DEFINE_integer(
    'num_steps', 1_000_000,
    'Number of environment steps to run the agent for.')

_BATCH_SIZE = 32
_SEQUENCE_LENGTH = _SEQUENCE_PERIOD = 20  # Avoids overlapping sequences.


def build_experiment_config():
  """Builds IMPALA experiment config which can be executed in different ways."""

  # Create an environment, grab the spec, and use it to create networks.
  env_name = ENV_NAME.value

  def env_factory(seed):
    del seed
    return helpers.make_atari_environment(
        level=env_name,
        sticky_actions=True,
        zero_discount_on_life_loss=False,
        oar_wrapper=True)

  # Construct the agent.
  num_learner_steps = NUM_ACTOR_STEPS.value // (_SEQUENCE_PERIOD * _BATCH_SIZE)
  lr_schedule = optax.linear_schedule(2e-4, 0., num_learner_steps)
  config = impala.IMPALAConfig(
      batch_size=_BATCH_SIZE,
      sequence_length=_SEQUENCE_LENGTH,
      sequence_period=_SEQUENCE_PERIOD,
      learning_rate=lr_schedule,
      entropy_cost=5e-3,
      max_abs_reward=1.,
  )

  return experiments.ExperimentConfig(
      builder=impala_builder.IMPALABuilder(config),
      environment_factory=env_factory,
      network_factory=impala.make_atari_networks,
      seed=SEED.value,
      max_num_actor_steps=NUM_ACTOR_STEPS.value)


def main(_):
  experiment_config = build_experiment_config()
  if RUN_DISTRIBUTED.value:
    program = experiments.make_distributed_experiment(
        experiment=experiment_config,
        num_actors=4 if lp_utils.is_local_run() else 256)
    lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
  else:
    experiments.run_experiment(experiment_config)


if __name__ == '__main__':
  app.run(main)
