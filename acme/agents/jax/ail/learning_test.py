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

"""Tests for the AIL learner."""
import functools

from acme import specs
from acme import types
from acme.agents.jax.ail import learning as ail_learning
from acme.agents.jax.ail import losses
from acme.agents.jax.ail import networks as ail_networks
from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import jax
import numpy as np
import optax

from absl.testing import absltest


def _make_discriminator(spec):
  def discriminator(*args, **kwargs) -> networks_lib.Logits:
    return ail_networks.DiscriminatorModule(
        environment_spec=spec,
        use_action=False,
        use_next_obs=False,
        network_core=ail_networks.DiscriminatorMLP([]))(*args, **kwargs)

  discriminator_transformed = hk.without_apply_rng(
      hk.transform_with_state(discriminator))
  return ail_networks.make_discriminator(
      environment_spec=spec,
      discriminator_transformed=discriminator_transformed)


class AilLearnerTest(absltest.TestCase):

  def test_step(self):
    simple_spec = specs.Array(shape=(), dtype=float)

    spec = specs.EnvironmentSpec(simple_spec, simple_spec, simple_spec,
                                 simple_spec)

    discriminator = _make_discriminator(spec)
    ail_network = ail_networks.AILNetworks(
        discriminator, imitation_reward_fn=lambda x: x, direct_rl_networks=None)

    loss = losses.gail_loss()

    optimizer = optax.adam(.01)

    step = jax.jit(functools.partial(
        ail_learning.ail_update_step,
        optimizer=optimizer,
        ail_network=ail_network,
        loss_fn=loss))

    zero_transition = types.Transition(
        np.array([0.]), np.array([0.]), 0., 0., np.array([0.]))
    zero_transition = utils.add_batch_dim(zero_transition)

    one_transition = types.Transition(
        np.array([1.]), np.array([0.]), 0., 0., np.array([0.]))
    one_transition = utils.add_batch_dim(one_transition)

    key = jax.random.PRNGKey(0)
    discriminator_params, discriminator_state = discriminator.init(key)

    state = ail_learning.DiscriminatorTrainingState(
        optimizer_state=optimizer.init(discriminator_params),
        discriminator_params=discriminator_params,
        discriminator_state=discriminator_state,
        policy_params=None,
        key=key,
        steps=0,
    )

    expected_loss = [1.062, 1.057, 1.052]

    for i in range(3):
      state, loss = step(state, (one_transition, zero_transition))
      self.assertAlmostEqual(loss['total_loss'], expected_loss[i], places=3)


if __name__ == '__main__':
  absltest.main()
