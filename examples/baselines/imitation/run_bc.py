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

"""Example running BC on continuous control tasks.

The network structure consists in a 3-layer MLP with ReLU activation
and dropout.
"""

from typing import Callable, Iterator, Tuple

from absl import flags
from acme import specs
from acme import types
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import bc
from acme.datasets import tfds
import helpers
from absl import app
from acme.jax import experiments
from acme.jax import types as jax_types
from acme.jax import utils
from acme.utils import lp_utils
import dm_env
import haiku as hk
import launchpad as lp
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'run_distributed', True, 'Should an agent be executed in a distributed '
    'way. If False, will run single-threaded.')
# Agent flags
flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'What environment to run')
flags.DEFINE_integer('num_demonstrations', 11,
                     'Number of demonstration trajectories.')
flags.DEFINE_integer('num_bc_steps', 100_000, 'Number of bc learning steps.')
flags.DEFINE_integer('num_steps', 0, 'Number of environment steps.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_float('learning_rate', 1e-4, 'Optimizer learning rate.')
flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate of bc network.')
flags.DEFINE_integer('num_layers', 3, 'Num layers of bc network.')
flags.DEFINE_integer('num_units', 256, 'Num units of bc network layers.')
flags.DEFINE_integer('eval_every', 5000, 'Evaluation period.')
flags.DEFINE_integer('evaluation_episodes', 10, 'Evaluation episodes.')
flags.DEFINE_integer('seed', 0, 'Random seed for learner and evaluator.')


def _make_demonstration_dataset_factory(
    dataset_name: str, num_demonstrations: int,
    environment_spec: specs.EnvironmentSpec, batch_size: int
) -> Callable[[jax_types.PRNGKey], Iterator[types.Transition]]:
  """Returns the demonstration dataset factory for the given dataset."""

  def demonstration_dataset_factory(
      random_key: jax_types.PRNGKey) -> Iterator[types.Transition]:
    """Returns an iterator of demonstration samples."""

    transitions_iterator = tfds.get_tfds_dataset(
        dataset_name, num_demonstrations, env_spec=environment_spec)
    return tfds.JaxInMemoryRandomSampleIterator(
        transitions_iterator, key=random_key, batch_size=batch_size)

  return demonstration_dataset_factory


def _make_environment_factory(env_name: str) -> jax_types.EnvironmentFactory:
  """Returns the environment factory for the given environment."""

  def environment_factory(seed: int) -> dm_env.Environment:
    del seed
    return helpers.make_environment(task=env_name)

  return environment_factory


def _make_network_factory(
    shift: Tuple[np.float64], scale: Tuple[np.float64], num_layers: int,
    num_units: int,
    dropout_rate: float) -> Callable[[specs.EnvironmentSpec], bc.BCNetworks]:
  """Returns the factory of networks to be used by the agent.

  Args:
    shift: Shift of the observations in demonstrations.
    scale: Scale of the observations in demonstrations.
    num_layers: Number of layers of the BC network.
    num_units: Number of units of the BC network.
    dropout_rate: Dropout rate of the BC network.

  Returns:
    Network factory.
  """

  def network_factory(spec: specs.EnvironmentSpec) -> bc.BCNetworks:
    """Creates the network used by the agent."""

    action_spec = spec.actions
    num_dimensions = np.prod(action_spec.shape, dtype=int)

    def actor_fn(obs, is_training=False, key=None):
      obs += shift
      obs *= scale
      hidden_layers = [num_units] * num_layers
      mlp = hk.Sequential([
          hk.nets.MLP(hidden_layers + [num_dimensions]),
      ])
      if is_training:
        return mlp(obs, dropout_rate=dropout_rate, rng=key)
      else:
        return mlp(obs)

    policy = hk.without_apply_rng(hk.transform(actor_fn))

    # Create dummy observations to create network parameters.
    dummy_obs = utils.zeros_like(spec.observations)
    dummy_obs = utils.add_batch_dim(dummy_obs)

    policy_network = bc.BCPolicyNetwork(lambda key: policy.init(key, dummy_obs),
                                        policy.apply)

    return bc.BCNetworks(policy_network=policy_network)

  return network_factory


def build_experiment_config() -> experiments.OfflineExperimentConfig[
    bc.BCNetworks, actor_core_lib.FeedForwardPolicy, types.Transition]:
  """Returns a config for BC experiments."""

  # Create an environment, grab the spec, and use it to create networks.
  environment = helpers.make_environment(task=FLAGS.env_name)
  environment_spec = specs.make_environment_spec(environment)

  # Define the demonstrations factory.
  dataset_name = helpers.get_dataset_name(FLAGS.env_name)
  demonstration_dataset_factory = _make_demonstration_dataset_factory(
      dataset_name, FLAGS.num_demonstrations, environment_spec,
      FLAGS.batch_size)

  # Load the demonstrations to compute the stats.
  dataset = tfds.get_tfds_dataset(
      dataset_name, FLAGS.num_demonstrations, env_spec=environment_spec)
  shift, scale = helpers.get_observation_stats(dataset)

  # Define the network factory.
  network_factory = _make_network_factory(  # pytype: disable=wrong-arg-types  # numpy-scalars
      shift=shift,
      scale=scale,
      num_layers=FLAGS.num_layers,
      num_units=FLAGS.num_units,
      dropout_rate=FLAGS.dropout_rate)

  # Create the BC builder.
  bc_config = bc.BCConfig(learning_rate=FLAGS.learning_rate)
  bc_builder = bc.BCBuilder(bc_config, loss_fn=bc.mse())

  environment_factory = _make_environment_factory(FLAGS.env_name)

  return experiments.OfflineExperimentConfig(
      builder=bc_builder,
      network_factory=network_factory,
      demonstration_dataset_factory=demonstration_dataset_factory,
      environment_factory=environment_factory,
      max_num_learner_steps=FLAGS.num_bc_steps,
      seed=FLAGS.seed,
      environment_spec=environment_spec,
  )


def main(_):
  config = build_experiment_config()
  if FLAGS.run_distributed:
    program = experiments.make_distributed_offline_experiment(experiment=config)
    lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program))
  else:
    experiments.run_offline_experiment(
        experiment=config,
        eval_every=FLAGS.eval_every,
        num_eval_episodes=FLAGS.evaluation_episodes)


if __name__ == '__main__':
  app.run(main)
