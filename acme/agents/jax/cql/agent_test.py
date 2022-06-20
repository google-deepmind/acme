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

"""Tests for the CQL agent."""

from acme import specs
from acme.agents.jax import cql
from acme.testing import fakes
import jax
import optax

from absl.testing import absltest


class CQLTest(absltest.TestCase):

  def test_train(self):
    seed = 0
    num_iterations = 6
    batch_size = 64

    # Create a fake environment to test with.
    environment = fakes.ContinuousEnvironment(
        episode_length=10, bounded=True, action_dim=6)
    spec = specs.make_environment_spec(environment)

    # Construct the agent.
    networks = cql.make_networks(
        spec, hidden_layer_sizes=(8, 8))
    dataset = fakes.transition_iterator(environment)
    key = jax.random.PRNGKey(seed)
    learner = cql.CQLLearner(
        batch_size,
        networks,
        key,
        demonstrations=dataset(batch_size),
        policy_optimizer=optax.adam(3e-5),
        critic_optimizer=optax.adam(3e-4),
        fixed_cql_coefficient=5.,
        cql_lagrange_threshold=None,
        target_entropy=0.1,
        num_bc_iters=2,
        num_sgd_steps_per_step=1)

    # Train the agent
    for _ in range(num_iterations):
      learner.step()


if __name__ == '__main__':
  absltest.main()
