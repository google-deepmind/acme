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

"""Tests for the SAC agent."""

from acme import specs
from acme.agents.jax import sac
from acme.testing import fakes
import jax
import optax

from absl.testing import absltest
from absl.testing import parameterized


class SACTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('adaptive_entropy', None, 0.),
      ('fixed_entropy', 0.2, 0.))
  def test_train(self, entropy_coefficient, target_entropy):
    seed = 0
    num_iterations = 5
    batch_size = 64

    environment = fakes.ContinuousEnvironment(
        episode_length=10, bounded=True, action_dim=6)
    spec = specs.make_environment_spec(environment)

    networks = sac.make_networks(spec, hidden_layer_sizes=(8, 8))
    dataset = fakes.transition_iterator(environment)
    key = jax.random.PRNGKey(seed)
    learner = sac.SACLearner(
        networks=networks,
        rng=key,
        iterator=dataset(batch_size),
        policy_optimizer=optax.adam(3e-4),
        q_optimizer=optax.adam(3e-4),
        entropy_coefficient=entropy_coefficient,
        target_entropy=target_entropy,
        num_sgd_steps_per_step=1)

    for _ in range(num_iterations):
      learner.step()


if __name__ == '__main__':
  absltest.main()
