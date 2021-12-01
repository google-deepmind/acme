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

"""Tests for the AIL agent."""
import functools
import shutil
from typing import Any, Callable

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import acme
from acme import specs
from acme import types
from acme.agents.jax import ail
from acme.agents.jax import ppo
from acme.agents.jax import sac
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax.layouts import local_layout
from acme.testing import fakes
from acme.utils import counting
from flax import linen
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np


NUM_DISCRETE_ACTIONS = 5
NUM_OBSERVATIONS = 10
OBS_SHAPE = (10, 5)
OBS_DTYPE = np.float32
EPISODE_LENGTH = 10
CONTINUOUS_ACTION_DIM = 3
CONTINUOUS_OBS_DIM = 5


def make_sac_logpi(
    sac_networks: sac.SACNetworks
) -> Callable[
    [networks_lib.Params, networks_lib.Observation, networks_lib.Action],
    jnp.ndarray]:
  """Returns the logpi function for SAC.

  Args:
    sac_networks: sac networks.

  Returns:
    A function from model params, obsevations, actions to log probability.
  """

  def logpi(params: networks_lib.Params, observations: networks_lib.Observation,
            actions: networks_lib.Action) -> jnp.ndarray:
    """Log probability of the action with the current policy."""
    model_output = sac_networks.policy_network.apply(params, observations)
    return sac_networks.log_prob(model_output, actions)

  return logpi


def ppo_forward_fn(inputs, num_actions):
  policy_network = hk.Sequential([
      utils.batch_concat,
      hk.nets.MLP([64, 64]),
      networks_lib.CategoricalHead(num_actions)
  ])
  value_network = hk.Sequential([
      utils.batch_concat,
      hk.nets.MLP([64, 64]),
      hk.Linear(1), lambda x: jnp.squeeze(x, axis=-1)
  ])

  action_distribution = policy_network(inputs)
  value = value_network(inputs)
  return (action_distribution, value)


def make_ppo_networks(
    spec: specs.EnvironmentSpec) -> networks_lib.FeedForwardNetwork:
  """Creates Haiku networks to be used by the agent."""

  num_actions = spec.actions.num_values

  forward_fn = functools.partial(ppo_forward_fn, num_actions=num_actions)

  # Transform into pure functions.
  forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

  dummy_obs = utils.zeros_like(spec.observations)
  dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.
  return networks_lib.FeedForwardNetwork(
      lambda rng: forward_fn.init(rng, dummy_obs), forward_fn.apply)


class DiscriminatorModule(linen.Module):
  """Discriminator module that concatenates its inputs."""
  # Environment spec
  environment_spec: specs.EnvironmentSpec
  # Network core
  network_core: Callable[..., Any]

  @linen.compact
  def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray,
               next_observations: jnp.ndarray, is_training: bool,
               rng: networks_lib.PRNGKey):
    output = self.network_core(observations)
    output = jnp.squeeze(output, axis=-1)
    return output


class AILTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('sac', 'sac'), ('ppo', 'ppo'), ('sac_airl', 'sac', True),
      ('sac_logpi', 'sac', False, True),
      ('sac_dropout', 'sac', False, False, .75),
      ('sac_spectral_norm', 'sac', False, True, 0., 1.))
  def test_ail(self,
               algo,
               airl_discriminator=False,
               subtract_logpi=False,
               dropout=0.,
               lipschitz_coeff=None):
    shutil.rmtree(flags.FLAGS.test_tmpdir, ignore_errors=True)
    batch_size = 8
    # Mujoco environment and associated demonstration dataset.
    if algo == 'ppo':
      environment = fakes.DiscreteEnvironment(
          num_actions=NUM_DISCRETE_ACTIONS,
          num_observations=NUM_OBSERVATIONS,
          obs_shape=OBS_SHAPE,
          obs_dtype=OBS_DTYPE,
          episode_length=EPISODE_LENGTH)
    else:
      environment = fakes.ContinuousEnvironment(
          episode_length=EPISODE_LENGTH,
          action_dim=CONTINUOUS_ACTION_DIM,
          observation_dim=CONTINUOUS_OBS_DIM,
          bounded=True)
    spec = specs.make_environment_spec(environment)

    if algo == 'sac':
      networks = sac.make_networks(spec=spec)
      config = sac.SACConfig(batch_size=batch_size,
                             samples_per_insert_tolerance_rate=float('inf'),
                             min_replay_size=1)
      base_builder = sac.SACBuilder(config=config)
      direct_rl_batch_size = batch_size
      behavior_policy = sac.apply_policy_and_sample(networks)
    elif algo == 'ppo':
      unroll_length = 5
      distribution_value_networks = make_ppo_networks(spec)
      networks = ppo.make_ppo_networks(distribution_value_networks)
      config = ppo.PPOConfig(
          unroll_length=unroll_length,
          num_minibatches=2,
          num_epochs=4,
          batch_size=batch_size)
      base_builder = ppo.PPOBuilder(config=config)
      direct_rl_batch_size = batch_size * unroll_length
      behavior_policy = jax.jit(ppo.make_inference_fn(networks),
                                backend='cpu')
    else:
      raise ValueError(f'Unexpected algorithm {algo}')

    if subtract_logpi:
      assert algo == 'sac'
      logpi_fn = make_sac_logpi(networks)
    else:
      logpi_fn = None

    if algo == 'ppo':
      embedding = lambda x: jnp.reshape(x, list(x.shape[:-2]) + [-1])
    else:
      embedding = lambda x: x

    def discriminator(*args, **kwargs) -> networks_lib.Logits:
      if airl_discriminator:
        return ail.AIRLModule(
            environment_spec=spec,
            use_action=True,
            use_next_obs=True,
            discount=.99,
            g_core=ail.DiscriminatorMLP(
                [4, 4],
                hidden_dropout_rate=dropout,
                spectral_normalization_lipschitz_coeff=lipschitz_coeff
            ),
            h_core=ail.DiscriminatorMLP(
                [4, 4],
                hidden_dropout_rate=dropout,
                spectral_normalization_lipschitz_coeff=lipschitz_coeff
            ),
            observation_embedding=embedding)(*args, **kwargs)
      else:
        return ail.DiscriminatorModule(
            environment_spec=spec,
            use_action=True,
            use_next_obs=True,
            network_core=ail.DiscriminatorMLP(
                [4, 4],
                hidden_dropout_rate=dropout,
                spectral_normalization_lipschitz_coeff=lipschitz_coeff
            ),
            observation_embedding=embedding)(*args, **kwargs)

    discriminator_transformed = hk.without_apply_rng(
        hk.transform_with_state(discriminator))

    discriminator_network = ail.make_discriminator(
        environment_spec=spec,
        discriminator_transformed=discriminator_transformed,
        logpi_fn=logpi_fn)

    networks = ail.AILNetworks(discriminator_network, lambda x: x, networks)

    builder = ail.AILBuilder(
        base_builder,
        config=ail.AILConfig(
            is_sequence_based=(algo == 'ppo'),
            share_iterator=True,
            direct_rl_batch_size=direct_rl_batch_size,
            discriminator_batch_size=2,
            policy_variable_name='policy' if subtract_logpi else None,
            min_replay_size=1),
        discriminator_loss=ail.losses.gail_loss(),
        make_demonstrations=fakes.transition_iterator(environment))

    # Construct the agent.
    agent = local_layout.LocalLayout(
        seed=0,
        environment_spec=spec,
        builder=builder,
        networks=networks,
        policy_network=behavior_policy,
        min_replay_size=1,
        batch_size=batch_size)

    # Train the agent.
    train_loop = acme.EnvironmentLoop(environment, agent)
    train_loop.run(num_episodes=(10 if algo == 'ppo' else 1))

  def test_ail_flax(self):
    shutil.rmtree(flags.FLAGS.test_tmpdir)
    batch_size = 8
    # Mujoco environment and associated demonstration dataset.
    environment = fakes.ContinuousEnvironment(
        episode_length=EPISODE_LENGTH,
        action_dim=CONTINUOUS_ACTION_DIM,
        observation_dim=CONTINUOUS_OBS_DIM,
        bounded=True)
    spec = specs.make_environment_spec(environment)

    networks = sac.make_networks(spec=spec)
    config = sac.SACConfig(batch_size=batch_size,
                           samples_per_insert_tolerance_rate=float('inf'),
                           min_replay_size=1)
    base_builder = sac.SACBuilder(config=config)
    direct_rl_batch_size = batch_size
    behavior_policy = sac.apply_policy_and_sample(networks)

    discriminator_module = DiscriminatorModule(spec, linen.Dense(1))

    def apply_fn(params: networks_lib.Params,
                 policy_params: networks_lib.Params, state: networks_lib.Params,
                 transitions: types.Transition, is_training: bool,
                 rng: networks_lib.PRNGKey) -> networks_lib.Logits:
      del policy_params
      variables = dict(params=params, **state)
      return discriminator_module.apply(
          variables,
          transitions.observation,
          transitions.action,
          transitions.next_observation,
          is_training=is_training,
          rng=rng,
          mutable=state.keys())

    def init_fn(rng):
      variables = discriminator_module.init(
          rng, dummy_obs, dummy_actions, dummy_obs, is_training=False, rng=rng)
      init_state, discriminator_params = variables.pop('params')
      return discriminator_params, init_state

    dummy_obs = utils.zeros_like(spec.observations)
    dummy_obs = utils.add_batch_dim(dummy_obs)
    dummy_actions = utils.zeros_like(spec.actions)
    dummy_actions = utils.add_batch_dim(dummy_actions)
    discriminator_network = networks_lib.FeedForwardNetwork(
        init=init_fn, apply=apply_fn)

    networks = ail.AILNetworks(discriminator_network, lambda x: x, networks)

    builder = ail.AILBuilder(
        base_builder,
        config=ail.AILConfig(
            is_sequence_based=False,
            share_iterator=True,
            direct_rl_batch_size=direct_rl_batch_size,
            discriminator_batch_size=2,
            policy_variable_name=None,
            min_replay_size=1),
        discriminator_loss=ail.losses.gail_loss(),
        make_demonstrations=fakes.transition_iterator(environment))

    counter = counting.Counter()
    # Construct the agent.
    agent = local_layout.LocalLayout(
        seed=0,
        environment_spec=spec,
        builder=builder,
        networks=networks,
        policy_network=behavior_policy,
        min_replay_size=1,
        batch_size=batch_size,
        counter=counter,
    )

    # Train the agent.
    train_loop = acme.EnvironmentLoop(environment, agent, counter=counter)
    train_loop.run(num_episodes=1)


if __name__ == '__main__':
  absltest.main()
