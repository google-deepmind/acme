# python3
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

"""Tests for PPO agent."""

from typing import Callable, Tuple

from absl.testing import absltest
import acme
from acme import specs
from acme.agents.jax import ppo
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.testing import fakes
from acme.utils import counting
import flax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability
tfd = tensorflow_probability.substrates.jax.distributions


def make_haiku_networks(
    spec: specs.EnvironmentSpec) -> networks_lib.FeedForwardNetwork:
  """Creates Haiku networks to be used by the agent."""

  num_actions = spec.actions.num_values

  def forward_fn(inputs):
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

  # Transform into pure functions.
  forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

  dummy_obs = utils.zeros_like(spec.observations)
  dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.
  return networks_lib.FeedForwardNetwork(
      lambda rng: forward_fn.init(rng, dummy_obs), forward_fn.apply)


def make_flax_networks(
    spec: specs.EnvironmentSpec) -> networks_lib.FeedForwardNetwork:
  """Creates FLAX networks to be used by the agent."""

  num_actions = spec.actions.num_values

  class MLP(flax.nn.Module):
    """MLP module."""

    def apply(self,
              data: jnp.ndarray,
              layer_sizes: Tuple[int],
              activation: Callable[[jnp.ndarray], jnp.ndarray] = flax.nn.relu,
              kernel_init: object = jax.nn.initializers.lecun_uniform(),
              activate_final: bool = False,
              bias: bool = True):
      hidden = data
      for i, hidden_size in enumerate(layer_sizes):
        hidden = flax.nn.Dense(
            hidden,
            hidden_size,
            name=f'hidden_{i}',
            kernel_init=kernel_init,
            bias=bias)
        if i != len(layer_sizes) - 1 or activate_final:
          hidden = activation(hidden)
      return hidden

  class PolicyValueModule(flax.nn.Module):
    """MLP module."""

    def apply(self, inputs: jnp.ndarray):
      inputs = utils.batch_concat(inputs)
      logits = MLP(inputs, [64, 64, num_actions])
      value = MLP(inputs, [64, 64, 1])
      value = jnp.squeeze(value, axis=-1)
      return tfd.Categorical(logits=logits), value

  dummy_obs = utils.zeros_like(spec.observations)
  dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.
  return networks_lib.FeedForwardNetwork(
      lambda rng: PolicyValueModule.init(rng, dummy_obs)[1],
      PolicyValueModule.call)


class PPOTest(absltest.TestCase):

  def test_ppo_with_haiku(self):
    self.run_ppo_agent(make_haiku_networks)

  def test_ppo_with_flax(self):
    self.run_ppo_agent(make_flax_networks)

  def run_ppo_agent(self, make_networks_fn):
    # Create a fake environment to test with.
    environment = fakes.DiscreteEnvironment(
        num_actions=5,
        num_observations=10,
        obs_shape=(10, 5),
        obs_dtype=np.float32,
        episode_length=10)
    spec = specs.make_environment_spec(environment)

    distribution_value_networks = make_networks_fn(spec)
    ppo_networks = ppo.make_ppo_networks(distribution_value_networks)
    config = ppo.PPOConfig(unroll_length=4, num_epochs=2, num_minibatches=2)
    workdir = self.create_tempdir()
    counter = counting.Counter()
    # Construct the agent.
    agent = ppo.PPO(
        spec=spec,
        networks=ppo_networks,
        config=config,
        seed=0,
        workdir=workdir.full_path,
        normalize_input=True,
        counter=counter,
    )

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent, counter=counter)
    loop.run(num_episodes=20)

  def test_ppo_nest_safety(self):
    # Create a fake environment with nested observations.
    environment = fakes.NestedDiscreteEnvironment(
        num_observations={
            'lat': 2,
            'long': 3
        },
        num_actions=5,
        obs_shape=(10, 5),
        obs_dtype=np.float32,
        episode_length=10)
    spec = specs.make_environment_spec(environment)

    distribution_value_networks = make_haiku_networks(spec)
    ppo_networks = ppo.make_ppo_networks(distribution_value_networks)
    config = ppo.PPOConfig(unroll_length=4, num_epochs=2, num_minibatches=2)
    workdir = self.create_tempdir()
    # Construct the agent.
    agent = ppo.PPO(
        spec=spec,
        networks=ppo_networks,
        config=config,
        seed=0,
        workdir=workdir.full_path,
        normalize_input=True,
    )

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent)
    loop.run(num_episodes=20)


if __name__ == '__main__':
  absltest.main()
