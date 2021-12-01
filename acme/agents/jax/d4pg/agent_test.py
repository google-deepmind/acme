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

"""Tests for the D4PG agent."""

from typing import Sequence

from absl.testing import absltest
import acme
from acme import specs
from acme.agents.jax import d4pg
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.testing import fakes
from acme.utils import counting
import haiku as hk
import jax.numpy as jnp
import numpy as np


def make_networks(
    spec: specs.EnvironmentSpec,
    policy_layer_sizes: Sequence[int] = (300, 200),
    critic_layer_sizes: Sequence[int] = (400, 300),
    vmin: float = -150.,
    vmax: float = 150.,
    num_atoms: int = 51,
) -> d4pg.D4PGNetworks:
  """Creates networks used by the agent."""

  action_spec = spec.actions

  num_dimensions = np.prod(action_spec.shape, dtype=int)
  critic_atoms = jnp.linspace(vmin, vmax, num_atoms)

  def _actor_fn(obs):
    network = hk.Sequential([
        utils.batch_concat,
        networks_lib.LayerNormMLP(policy_layer_sizes, activate_final=True),
        networks_lib.NearZeroInitializedLinear(num_dimensions),
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


class D4PGTest(absltest.TestCase):

  def test_d4pg(self):
    # Create a fake environment to test with.
    environment = fakes.ContinuousEnvironment(
        episode_length=10, action_dim=3, observation_dim=5, bounded=True)
    spec = specs.make_environment_spec(environment)

    # Create the networks.
    networks = make_networks(spec)

    config = d4pg.D4PGConfig(
        batch_size=10,
        samples_per_insert=2,
        min_replay_size=10,
        samples_per_insert_tolerance_rate=float('inf'))
    counter = counting.Counter()
    agent = d4pg.D4PG(spec, networks, config=config, random_seed=0,
                      counter=counter)

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent, counter=counter)
    loop.run(num_episodes=2)


if __name__ == '__main__':
  absltest.main()
