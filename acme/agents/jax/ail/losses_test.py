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

"""Tests for the AIL discriminator losses."""

from acme import types
from acme.agents.jax.ail import losses
from acme.jax import networks as networks_lib
import jax
import jax.numpy as jnp
import tree

from absl.testing import absltest


class AilLossTest(absltest.TestCase):

  def test_gradient_penalty(self):

    def dummy_discriminator(
        transition: types.Transition) -> networks_lib.Logits:
      return transition.observation + jnp.square(transition.action)

    zero_transition = types.Transition(0., 0., 0., 0., 0.)
    zero_transition = tree.map_structure(lambda x: jnp.expand_dims(x, axis=0),
                                         zero_transition)
    self.assertEqual(
        losses._compute_gradient_penalty(zero_transition, dummy_discriminator,
                                         0.), 1**2 + 0**2)

    one_transition = types.Transition(1., 1., 0., 0., 0.)
    one_transition = tree.map_structure(lambda x: jnp.expand_dims(x, axis=0),
                                        one_transition)
    self.assertEqual(
        losses._compute_gradient_penalty(one_transition, dummy_discriminator,
                                         0.), 1**2 + 2**2)

  def test_pugail(self):

    def dummy_discriminator(
        state: losses.State,
        transition: types.Transition) -> losses.DiscriminatorOutput:
      return transition.observation, state

    zero_transition = types.Transition(.1, 0., 0., 0., 0.)
    zero_transition = tree.map_structure(lambda x: jnp.expand_dims(x, axis=0),
                                         zero_transition)

    one_transition = types.Transition(1., 0., 0., 0., 0.)
    one_transition = tree.map_structure(lambda x: jnp.expand_dims(x, axis=0),
                                        one_transition)

    prior = .7
    loss_fn = losses.pugail_loss(
        positive_class_prior=prior, entropy_coefficient=0.)
    loss, _ = loss_fn(dummy_discriminator, {}, one_transition,
                      zero_transition, ())

    d_one = jax.nn.sigmoid(dummy_discriminator({}, one_transition)[0])
    d_zero = jax.nn.sigmoid(dummy_discriminator({}, zero_transition)[0])
    expected_loss = -prior * jnp.log(
        d_one) + -jnp.log(1. - d_zero) - prior * -jnp.log(1 - d_one)

    self.assertAlmostEqual(loss, expected_loss, places=6)


if __name__ == '__main__':
  absltest.main()
