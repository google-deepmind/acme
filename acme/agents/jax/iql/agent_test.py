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

"""Tests for IQL agent."""

from absl.testing import absltest
from acme import specs
from acme.agents.jax import iql
from acme.testing import fakes
from acme.utils import counting
from acme.utils import loggers
import jax
import numpy as np


class IQLTest(absltest.TestCase):
  """Basic tests for IQL agent components."""

  def test_iql_networks_creation(self):
    """Test that IQL networks can be created."""
    # Create a simple environment spec
    env = fakes.ContinuousEnvironment(
        episode_length=10,
        action_dim=2,
        observation_dim=4,
        bounded=True)
    env_spec = specs.make_environment_spec(env)
    
    # Create networks
    networks = iql.make_networks(env_spec)
    
    # Check that all networks are created
    self.assertIsNotNone(networks.policy_network)
    self.assertIsNotNone(networks.q_network)
    self.assertIsNotNone(networks.value_network)
    self.assertIsNotNone(networks.log_prob)
    self.assertIsNotNone(networks.sample)
    self.assertIsNotNone(networks.sample_eval)

  def test_iql_config(self):
    """Test IQL config creation with default values."""
    config = iql.IQLConfig()
    
    self.assertEqual(config.batch_size, 256)
    self.assertEqual(config.expectile, 0.7)
    self.assertEqual(config.temperature, 3.0)
    self.assertEqual(config.discount, 0.99)

  def test_iql_builder(self):
    """Test that IQL builder can be created."""
    config = iql.IQLConfig(batch_size=64)
    builder = iql.IQLBuilder(config)
    
    self.assertIsNotNone(builder)

  def test_iql_learner_creation(self):
    """Test that IQL learner can be created and run."""
    # Create environment
    env = fakes.ContinuousEnvironment(
        episode_length=10,
        action_dim=2,
        observation_dim=4,
        bounded=True)
    env_spec = specs.make_environment_spec(env)
    
    # Create networks
    networks = iql.make_networks(env_spec)
    
    # Create fake dataset
    dataset = fakes.transition_iterator(env)(batch_size=32)
    
    # Create learner
    config = iql.IQLConfig(batch_size=32)
    learner = iql.IQLLearner(
        batch_size=config.batch_size,
        networks=networks,
        random_key=jax.random.PRNGKey(0),
        demonstrations=dataset,
        policy_optimizer=iql.optax.adam(config.policy_learning_rate),
        value_optimizer=iql.optax.adam(config.value_learning_rate),
        critic_optimizer=iql.optax.adam(config.critic_learning_rate),
        tau=config.tau,
        expectile=config.expectile,
        temperature=config.temperature,
        discount=config.discount,
        counter=counting.Counter(),
        logger=loggers.NoOpLogger())
    
    # Run a few training steps
    for _ in range(5):
      learner.step()
    
    # Check that parameters can be retrieved
    policy_params = learner.get_variables(['policy'])[0]
    self.assertIsNotNone(policy_params)


if __name__ == '__main__':
  absltest.main()
