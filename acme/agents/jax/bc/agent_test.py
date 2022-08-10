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

"""Tests for the BC agent."""

from acme import specs
from acme import types
from acme.agents.jax import bc
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.jax import utils
from acme.testing import fakes
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax.scipy import special
import numpy as np
import optax
import rlax

from absl.testing import absltest
from absl.testing import parameterized


def make_networks(spec: specs.EnvironmentSpec,
                  discrete_actions: bool = False) -> bc.BCNetworks:
  """Creates networks used by the agent."""

  if discrete_actions:
    final_layer_size = spec.actions.num_values
  else:
    final_layer_size = np.prod(spec.actions.shape, dtype=int)

  def _actor_fn(obs, is_training=False, key=None):
    # is_training and key allows to defined train/test dependant modules
    # like dropout.
    del is_training
    del key
    if discrete_actions:
      network = hk.nets.MLP([64, 64, final_layer_size])
    else:
      network = hk.Sequential([
          networks_lib.LayerNormMLP([64, 64], activate_final=True),
          networks_lib.NormalTanhDistribution(final_layer_size),
      ])
    return network(obs)

  policy = hk.without_apply_rng(hk.transform(_actor_fn))

  # Create dummy observations and actions to create network parameters.
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_obs = utils.add_batch_dim(dummy_obs)
  policy_network = networks_lib.FeedForwardNetwork(
      lambda key: policy.init(key, dummy_obs), policy.apply)
  bc_policy_network = bc.convert_to_bc_network(policy_network)

  if discrete_actions:

    def sample_fn(logits: networks_lib.NetworkOutput,
                  key: jax_types.PRNGKey) -> networks_lib.Action:
      return rlax.epsilon_greedy(epsilon=0.0).sample(key, logits)

    def log_prob(logits: networks_lib.NetworkOutput,
                 actions: networks_lib.Action) -> networks_lib.LogProb:
      max_logits = jnp.max(logits, axis=-1, keepdims=True)
      logits = logits - max_logits
      logits_actions = jnp.sum(
          jax.nn.one_hot(actions, spec.actions.num_values) * logits, axis=-1)

      log_prob = logits_actions - special.logsumexp(logits, axis=-1)
      return log_prob

  else:

    def sample_fn(distribution: networks_lib.NetworkOutput,
                  key: jax_types.PRNGKey) -> networks_lib.Action:
      return distribution.sample(seed=key)

    def log_prob(distribuition: networks_lib.NetworkOutput,
                 actions: networks_lib.Action) -> networks_lib.LogProb:
      return distribuition.log_prob(actions)

  return bc.BCNetworks(bc_policy_network, sample_fn, log_prob)


class BCTest(parameterized.TestCase):

  @parameterized.parameters(
      ('logp',),
      ('mse',),
      ('peerbc',)
      )
  def test_continuous_actions(self, loss_name):
    with chex.fake_pmap_and_jit():
      num_sgd_steps_per_step = 1
      num_steps = 5

      # Create a fake environment to test with.
      environment = fakes.ContinuousEnvironment(
          episode_length=10, bounded=True, action_dim=6)

      spec = specs.make_environment_spec(environment)
      dataset_demonstration = fakes.transition_dataset(environment)
      dataset_demonstration = dataset_demonstration.map(
          lambda sample: types.Transition(*sample.data))
      dataset_demonstration = dataset_demonstration.batch(8).as_numpy_iterator()

      # Construct the agent.
      networks = make_networks(spec)

      if loss_name == 'logp':
        loss_fn = bc.logp()
      elif loss_name == 'mse':
        loss_fn = bc.mse()
      elif loss_name == 'peerbc':
        loss_fn = bc.peerbc(bc.logp(), zeta=0.1)
      else:
        raise ValueError

      learner = bc.BCLearner(
          networks=networks,
          random_key=jax.random.PRNGKey(0),
          loss_fn=loss_fn,
          optimizer=optax.adam(0.01),
          prefetching_iterator=utils.sharded_prefetch(dataset_demonstration),
          num_sgd_steps_per_step=num_sgd_steps_per_step)

      # Train the agent
      for _ in range(num_steps):
        learner.step()

  @parameterized.parameters(
      ('logp',),
      ('rcal',))
  def test_discrete_actions(self, loss_name):
    with chex.fake_pmap_and_jit():

      num_sgd_steps_per_step = 1
      num_steps = 5

      # Create a fake environment to test with.
      environment = fakes.DiscreteEnvironment(
          num_actions=10, num_observations=100, obs_shape=(10,),
          obs_dtype=np.float32)

      spec = specs.make_environment_spec(environment)
      dataset_demonstration = fakes.transition_dataset(environment)
      dataset_demonstration = dataset_demonstration.map(
          lambda sample: types.Transition(*sample.data))
      dataset_demonstration = dataset_demonstration.batch(8).as_numpy_iterator()

      # Construct the agent.
      networks = make_networks(spec, discrete_actions=True)

      if loss_name == 'logp':
        loss_fn = bc.logp()

      elif loss_name == 'rcal':
        base_loss_fn = bc.logp()
        loss_fn = bc.rcal(base_loss_fn, discount=0.99, alpha=0.1)

      else:
        raise ValueError

      learner = bc.BCLearner(
          networks=networks,
          random_key=jax.random.PRNGKey(0),
          loss_fn=loss_fn,
          optimizer=optax.adam(0.01),
          prefetching_iterator=utils.sharded_prefetch(dataset_demonstration),
          num_sgd_steps_per_step=num_sgd_steps_per_step)

      # Train the agent
      for _ in range(num_steps):
        learner.step()


if __name__ == '__main__':
  absltest.main()
