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

"""Tests for bc_initialization."""

from acme import specs
from acme.agents.jax import bc
from acme.agents.jax import sac
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.testing import fakes
import haiku as hk
import jax
import numpy as np

from absl.testing import absltest


def make_networks(spec: specs.EnvironmentSpec) -> bc.BCNetworks:
  """Creates networks used by the agent."""

  final_layer_size = np.prod(spec.actions.shape, dtype=int)

  def _actor_fn(obs, is_training=False, key=None):
    # is_training and key allows to defined train/test dependant modules
    # like dropout.
    del is_training
    del key
    network = networks_lib.LayerNormMLP([64, 64, final_layer_size],
                                        activate_final=False)
    return jax.nn.tanh(network(obs))

  policy = hk.without_apply_rng(hk.transform(_actor_fn))

  # Create dummy observations and actions to create network parameters.
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_obs = utils.add_batch_dim(dummy_obs)
  policy_network = bc.BCPolicyNetwork(lambda key: policy.init(key, dummy_obs),
                                      policy.apply)

  return bc.BCNetworks(policy_network)


class BcPretrainingTest(absltest.TestCase):

  def test_bc_initialization(self):
    # Create a fake environment to test with.
    environment = fakes.ContinuousEnvironment(
        episode_length=10, bounded=True, action_dim=6)
    spec = specs.make_environment_spec(environment)

    # Construct the agent.
    nets = make_networks(spec)

    loss = bc.mse()

    bc.pretraining.train_with_bc(
        fakes.transition_iterator(environment), nets, loss, num_steps=100)

  def test_sac_to_bc_networks(self):
    # Create a fake environment to test with.
    environment = fakes.ContinuousEnvironment(
        episode_length=10, bounded=True, action_dim=6)
    spec = specs.make_environment_spec(environment)

    sac_nets = sac.make_networks(spec, hidden_layer_sizes=(4, 4))
    bc_nets = bc.convert_to_bc_network(sac_nets.policy_network)

    rng = jax.random.PRNGKey(0)
    dummy_obs = utils.zeros_like(spec.observations)
    dummy_obs = utils.add_batch_dim(dummy_obs)

    sac_params = sac_nets.policy_network.init(rng)
    sac_output = sac_nets.policy_network.apply(sac_params, dummy_obs)

    bc_params = bc_nets.init(rng)
    bc_output = bc_nets.apply(bc_params, dummy_obs, is_training=False, key=None)

    np.testing.assert_array_equal(sac_output.mode(), bc_output.mode())


if __name__ == '__main__':
  absltest.main()
