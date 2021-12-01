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

from absl.testing import absltest
from absl.testing import parameterized
from acme import specs
from acme import types
from acme.agents.jax import bc
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.testing import fakes
import chex
import haiku as hk
import jax
import jax.numpy as jnp
from jax.scipy import special
import numpy as np
import optax


def make_networks(
    spec: specs.EnvironmentSpec,
    discrete_actions: bool = False) -> networks_lib.FeedForwardNetwork:
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
  network = networks_lib.FeedForwardNetwork(
      lambda key: policy.init(key, dummy_obs), policy.apply)
  return network


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
      network = make_networks(spec)

      if loss_name == 'logp':
        loss_fn = bc.logp(
            logp_fn=lambda dist_params, actions: dist_params.log_prob(actions))
      elif loss_name == 'mse':
        loss_fn = bc.mse(
            sample_fn=lambda dist_params, key: dist_params.sample(seed=key))
      elif loss_name == 'peerbc':
        base_loss_fn = bc.logp(
            logp_fn=lambda dist_params, actions: dist_params.log_prob(actions))
        loss_fn = bc.peerbc(base_loss_fn, zeta=0.1)
      else:
        raise ValueError

      learner = bc.BCLearner(
          network=network,
          random_key=jax.random.PRNGKey(0),
          loss_fn=loss_fn,
          optimizer=optax.adam(0.01),
          demonstrations=dataset_demonstration,
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
      network = make_networks(spec, discrete_actions=True)

      def logp_fn(logits, actions):
        max_logits = jnp.max(logits, axis=-1, keepdims=True)
        logits = logits - max_logits
        logits_actions = jnp.sum(
            jax.nn.one_hot(actions, spec.actions.num_values) * logits, axis=-1)

        log_prob = logits_actions - special.logsumexp(logits, axis=-1)
        return log_prob

      if loss_name == 'logp':
        loss_fn = bc.logp(logp_fn=logp_fn)

      elif loss_name == 'rcal':
        base_loss_fn = bc.logp(logp_fn=logp_fn)
        loss_fn = bc.rcal(base_loss_fn, discount=0.99, alpha=0.1)

      else:
        raise ValueError

      learner = bc.BCLearner(
          network=network,
          random_key=jax.random.PRNGKey(0),
          loss_fn=loss_fn,
          optimizer=optax.adam(0.01),
          demonstrations=dataset_demonstration,
          num_sgd_steps_per_step=num_sgd_steps_per_step)

      # Train the agent
      for _ in range(num_steps):
        learner.step()


if __name__ == '__main__':
  absltest.main()
