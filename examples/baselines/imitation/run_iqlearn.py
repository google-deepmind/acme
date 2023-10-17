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

"""Example running IQLearn on continuous control tasks.

This handles the online imitation setting.
"""

from typing import Callable, Iterator

from absl import flags
from acme import specs
from acme import types
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import iq_learn
from acme.datasets import tfds
import helpers
from absl import app
from acme.jax import experiments
from acme.jax import types as jax_types
from acme.utils import lp_utils
import dm_env
import jax
import launchpad as lp

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'run_distributed',
    True,
    (
        'Should an agent be executed in a distributed '
        'way. If False, will run single-threaded.'
    ),
)
flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'What environment to run')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('num_steps', 1_000_000, 'Number of env steps to run.')
flags.DEFINE_integer('eval_every', 50_000, 'Number of env steps to run.')
flags.DEFINE_integer(
    'num_demonstrations', 11, 'Number of demonstration trajectories.'
)
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')


def _make_environment_factory(env_name: str) -> jax_types.EnvironmentFactory:
  """Returns the environment factory for the given environment."""

  def environment_factory(seed: int) -> dm_env.Environment:
    del seed
    return helpers.make_environment(task=env_name)

  return environment_factory


def _make_demonstration_dataset_factory(
    dataset_name: str,
    environment_spec: specs.EnvironmentSpec,
    num_demonstrations: int,
    random_key: jax_types.PRNGKey,
) -> Callable[[jax_types.PRNGKey], Iterator[types.Transition]]:
  """Returns the demonstration dataset factory for the given dataset."""

  def demonstration_dataset_factory(
      batch_size: int,
  ) -> Iterator[types.Transition]:
    """Returns an iterator of demonstration samples."""
    transitions_iterator = tfds.get_tfds_dataset(
        dataset_name, num_episodes=num_demonstrations, env_spec=environment_spec
    )
    return tfds.JaxInMemoryRandomSampleIterator(
        transitions_iterator, key=random_key, batch_size=batch_size
    )

  return demonstration_dataset_factory  # pytype: disable=bad-return-type


def build_experiment_config() -> (
    experiments.ExperimentConfig[
        iq_learn.IQLearnNetworks,
        actor_core_lib.ActorCore,
        iq_learn.IQLearnSample,
    ]
):
  """Returns a configuration for IQLearn experiments."""

  # Create an environment, grab the spec, and use it to create networks.
  env_name = FLAGS.env_name
  environment_factory = _make_environment_factory(env_name)

  dummy_seed = 1
  environment = environment_factory(dummy_seed)
  environment_spec = specs.make_environment_spec(environment)

  # Create demonstrations function.
  dataset_name = helpers.get_dataset_name(env_name)
  make_demonstrations = _make_demonstration_dataset_factory(
      dataset_name,
      environment_spec,
      FLAGS.num_demonstrations,
      jax.random.PRNGKey(FLAGS.seed),
  )

  # Construct the agent
  iq_learn_config = iq_learn.IQLearnConfig(alpha=1.0)
  iq_learn_builder = iq_learn.IQLearnBuilder(  # pytype: disable=wrong-arg-types
      config=iq_learn_config, make_demonstrations=make_demonstrations
  )

  return experiments.ExperimentConfig(
      builder=iq_learn_builder,
      environment_factory=environment_factory,
      network_factory=iq_learn.make_networks,
      seed=FLAGS.seed,
      max_num_actor_steps=FLAGS.num_steps,
  )


def main(_):
  config = build_experiment_config()
  if FLAGS.run_distributed:
    program = experiments.make_distributed_experiment(
        experiment=config, num_actors=4
    )
    lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
  else:
    experiments.run_experiment(
        experiment=config,
        eval_every=FLAGS.eval_every,
        num_eval_episodes=FLAGS.evaluation_episodes,
    )


if __name__ == '__main__':
  app.run(main)
