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

"""Example running SQIL on continuous control tasks.

SQIL: Imitation Learning via Reinforcement Learning with Sparse Rewards
Reddy et al., 2019 https://arxiv.org/abs/1905.11108
"""

from absl import flags
from acme import specs
from acme.agents.jax import sac
from acme.agents.jax import sqil
from acme.datasets import tfds
import helpers
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import dm_env
import jax
import launchpad as lp


FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'What environment to run')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_steps', 1_000_000, 'Number of env steps to run.')
flags.DEFINE_integer('eval_every', 50_000, 'Number of env steps to run.')
flags.DEFINE_integer('num_demonstrations', 11,
                     'Number of demonstration trajectories.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')


def build_experiment_config() -> experiments.ExperimentConfig:
  """Returns a configuration for SQIL experiments."""

  # Create an environment, grab the spec, and use it to create networks.
  env_name = FLAGS.env_name

  def environment_factory(seed: int) -> dm_env.Environment:
    del seed
    return helpers.make_environment(task=env_name)

  dummy_seed = 1
  environment = environment_factory(dummy_seed)
  environment_spec = specs.make_environment_spec(environment)

  # Construct the agent.
  sac_config = sac.SACConfig(
      target_entropy=sac.target_entropy_from_env_spec(environment_spec),
      min_replay_size=1,
      samples_per_insert_tolerance_rate=2.0)
  sac_builder = sac.SACBuilder(sac_config)

  # Create demonstrations function.
  dataset_name = helpers.get_dataset_name(FLAGS.env_name)
  num_demonstrations = FLAGS.num_demonstrations
  def make_demonstrations(batch_size: int, seed: int = 0):
    transitions_iterator = tfds.get_tfds_dataset(
        dataset_name, num_demonstrations, env_spec=environment_spec)
    return tfds.JaxInMemoryRandomSampleIterator(
        transitions_iterator, jax.random.PRNGKey(seed), batch_size)

  sqil_builder = sqil.SQILBuilder(sac_builder, sac_config.batch_size,
                                  make_demonstrations)

  return experiments.ExperimentConfig(
      builder=sqil_builder,
      environment_factory=environment_factory,
      network_factory=sac.make_networks,
      seed=FLAGS.seed,
      max_num_actor_steps=FLAGS.num_steps)


def main(_):
  config = build_experiment_config()
  if FLAGS.run_distributed:
    program = experiments.make_distributed_experiment(
        experiment=config, num_actors=4)
    lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
  else:
    experiments.run_experiment(
        experiment=config,
        eval_every=FLAGS.eval_every,
        num_eval_episodes=FLAGS.evaluation_episodes)


if __name__ == '__main__':
  app.run(main)
