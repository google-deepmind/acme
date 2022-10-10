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

"""Example running PWIL on continuous control tasks.

The network structure and hyperparameters are the same as the one used in the
PWIL paper: https://arxiv.org/pdf/2006.04678.pdf.
"""

from typing import Sequence

from absl import flags
from acme import specs
from acme.agents.jax import d4pg
from acme.agents.jax import pwil
from acme.datasets import tfds
import helpers
from absl import app
from acme.jax import experiments
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import lp_utils
import dm_env
import haiku as hk
import jax.numpy as jnp
import launchpad as lp
import numpy as np


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


def make_networks(
    spec: specs.EnvironmentSpec,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    vmin: float = -150.,
    vmax: float = 150.,
    num_atoms: int = 201,
) -> d4pg.D4PGNetworks:
  """Creates networks used by the agent."""

  action_spec = spec.actions

  num_dimensions = np.prod(action_spec.shape, dtype=int)
  critic_atoms = jnp.linspace(vmin, vmax, num_atoms)

  def _actor_fn(obs):
    network = hk.Sequential([
        utils.batch_concat,
        networks_lib.LayerNormMLP(list(policy_layer_sizes) + [num_dimensions]),
        networks_lib.TanhToSpec(action_spec),
    ])
    return network(obs)

  def _critic_fn(obs, action):
    network = hk.Sequential([
        utils.batch_concat,
        networks_lib.LayerNormMLP(layer_sizes=[*critic_layer_sizes, num_atoms]),
    ])
    value = network([obs, action])
    return value, critic_atoms

  policy = hk.without_apply_rng(hk.transform(_actor_fn))
  critic = hk.without_apply_rng(hk.transform(_critic_fn))

  # Create dummy observations and actions to create network parameters.
  dummy_action = utils.zeros_like(spec.actions)
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_action = utils.add_batch_dim(dummy_action)
  dummy_obs = utils.add_batch_dim(dummy_obs)

  return d4pg.D4PGNetworks(
      policy_network=networks_lib.FeedForwardNetwork(
          lambda rng: policy.init(rng, dummy_obs), policy.apply),
      critic_network=networks_lib.FeedForwardNetwork(
          lambda rng: critic.init(rng, dummy_obs, dummy_action), critic.apply))


def build_experiment_config() -> experiments.ExperimentConfig:
  """Returns a configuration for PWIL experiments."""

  # Create an environment, grab the spec, and use it to create networks.
  env_name = FLAGS.env_name

  def environment_factory(seed: int) -> dm_env.Environment:
    del seed
    return helpers.make_environment(task=env_name)

  dummy_seed = 1
  environment = environment_factory(dummy_seed)
  environment_spec = specs.make_environment_spec(environment)

  # Create d4pg agent
  d4pg_config = d4pg.D4PGConfig(
      learning_rate=5e-5, sigma=0.2, samples_per_insert=256)
  d4pg_builder = d4pg.D4PGBuilder(config=d4pg_config)

  # Create demonstrations function.
  dataset_name = helpers.get_dataset_name(FLAGS.env_name)
  num_demonstrations = FLAGS.num_demonstrations

  def make_demonstrations():
    transitions_iterator = tfds.get_tfds_dataset(
        dataset_name, num_demonstrations, env_spec=environment_spec)
    return pwil.PWILDemonstrations(
        demonstrations=transitions_iterator, episode_length=1000)

  # Construct PWIL agent
  pwil_config = pwil.PWILConfig(num_transitions_rb=0)
  pwil_builder = pwil.PWILBuilder(
      rl_agent=d4pg_builder,
      config=pwil_config,
      demonstrations_fn=make_demonstrations)

  return experiments.ExperimentConfig(
      builder=pwil_builder,
      environment_factory=environment_factory,
      network_factory=make_networks,
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
