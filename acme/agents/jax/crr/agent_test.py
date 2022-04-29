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

"""Tests for the CRR agent."""

from acme import specs
from acme.agents.jax import crr
from acme.testing import fakes
import jax
import optax

from absl.testing import absltest
from absl.testing import parameterized


class CRRTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('exp', crr.policy_loss_coeff_advantage_exp),
      ('indicator', crr.policy_loss_coeff_advantage_indicator),
      ('all', crr.policy_loss_coeff_constant))
  def test_train(self, policy_loss_coeff_fn):
    seed = 0
    num_iterations = 5
    batch_size = 64
    grad_updates_per_batch = 1

    # Create a fake environment to test with.
    environment = fakes.ContinuousEnvironment(
        episode_length=10, bounded=True, action_dim=6)
    spec = specs.make_environment_spec(environment)

    # Construct the learner.
    networks = crr.make_networks(
        spec, policy_layer_sizes=(8, 8), critic_layer_sizes=(8, 8))
    key = jax.random.PRNGKey(seed)
    dataset = fakes.transition_iterator(environment)
    learner = crr.CRRLearner(
        networks,
        key,
        discount=0.95,
        target_update_period=2,
        policy_loss_coeff_fn=policy_loss_coeff_fn,
        iterator=dataset(batch_size * grad_updates_per_batch),
        policy_optimizer=optax.adam(1e-4),
        critic_optimizer=optax.adam(1e-4),
        grad_updates_per_batch=grad_updates_per_batch)

    # Train the learner.
    for _ in range(num_iterations):
      learner.step()


if __name__ == '__main__':
  absltest.main()
