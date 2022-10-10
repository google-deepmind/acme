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

"""Example running GAIL/DAC on continuous control tasks.

GAIL: Generative Adversarial Imitation Learning
Ho & Ermon, 2016 https://arxiv.org/abs/1606.03476

DAC: Discriminator Actor-Critic: Addressing Sample Inefficiency And Reward
Bias In Adversarial Imitation Learning
Kostrikov et al., 2018 https://arxiv.org/pdf/1809.02925.pdf

We use TD3 similarly to DAC and do not use the extra absorbing state described
in DAC and use a different reward that corresponds to GAIL.

The network structure and hyperparameters of the discriminator are the ones
defined in the following paper:
What Matters in Adversarial Reinforcement Learning, Orsini et al., 2021
https://arxiv.org/pdf/2106.00672.pdf.

The changes lead to an improved agent able to learn from a single demonstration
(even for Humanoid).
"""

from absl import flags
from acme import specs
from acme.agents.jax import ail
from acme.agents.jax import td3
from acme.datasets import tfds
import helpers
from absl import app
from acme.jax import experiments
from acme.jax import networks as networks_lib
from acme.utils import lp_utils
import dm_env
import haiku as hk
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
  """Returns a configuration for GAIL/DAC experiments."""

  # Create an environment, grab the spec, and use it to create networks.
  environment = helpers.make_environment(task=FLAGS.env_name)
  environment_spec = specs.make_environment_spec(environment)

  # Create the direct RL agent.
  td3_config = td3.TD3Config(
      min_replay_size=1,
      samples_per_insert_tolerance_rate=2.0)
  td3_networks = td3.make_networks(environment_spec)

  # Create the discriminator.
  def discriminator(*args, **kwargs) -> networks_lib.Logits:
    return ail.DiscriminatorModule(
        environment_spec=environment_spec,
        use_action=True,
        use_next_obs=False,
        network_core=ail.DiscriminatorMLP(
            hidden_layer_sizes=[64,],
            spectral_normalization_lipschitz_coeff=1.)
        )(*args, **kwargs)
  discriminator_transformed = hk.without_apply_rng(
      hk.transform_with_state(discriminator))

  def network_factory(
      environment_spec: specs.EnvironmentSpec) -> ail.AILNetworks:
    return ail.AILNetworks(
        ail.make_discriminator(environment_spec, discriminator_transformed),
        # reward balance = 0 corresponds to the GAIL reward: -ln(1-D)
        imitation_reward_fn=ail.rewards.gail_reward(reward_balance=0.),
        direct_rl_networks=td3_networks)

  # Create demonstrations function.
  dataset_name = helpers.get_dataset_name(FLAGS.env_name)
  num_demonstrations = FLAGS.num_demonstrations
  def make_demonstrations(batch_size, seed: int = 0):
    transitions_iterator = tfds.get_tfds_dataset(
        dataset_name, num_demonstrations, env_spec=environment_spec)
    return tfds.JaxInMemoryRandomSampleIterator(
        transitions_iterator, jax.random.PRNGKey(seed), batch_size)

  # Create DAC agent.
  ail_config = ail.AILConfig(direct_rl_batch_size=td3_config.batch_size *
                             td3_config.num_sgd_steps_per_step)

  env_name = FLAGS.env_name

  def environment_factory(seed: int) -> dm_env.Environment:
    del seed
    return helpers.make_environment(task=env_name)

  td3_builder = td3.TD3Builder(td3_config)

  dac_loss = ail.losses.add_gradient_penalty(
      ail.losses.gail_loss(entropy_coefficient=1e-3),
      gradient_penalty_coefficient=10.,
      gradient_penalty_target=1.)

  ail_builder = ail.AILBuilder(
      rl_agent=td3_builder,
      config=ail_config,
      discriminator_loss=dac_loss,
      make_demonstrations=make_demonstrations)

  return experiments.ExperimentConfig(
      builder=ail_builder,
      environment_factory=environment_factory,
      network_factory=network_factory,
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
