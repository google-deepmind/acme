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

"""Tests for the IMPALA loss function."""

from acme.adders import reverb as adders
from acme.jax.losses import impala
from acme.utils.tree_utils import tree_map
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import reverb

from absl.testing import absltest


class ImpalaTest(absltest.TestCase):

  def test_shapes(self):

    #
    batch_size = 2
    sequence_len = 3
    num_actions = 5
    hidden_size = 7

    # Define a trivial recurrent actor-critic network.
    @hk.without_apply_rng
    @hk.transform
    def unroll_fn_transformed(observations, state):
      lstm = hk.LSTM(hidden_size)
      embedding, state = hk.dynamic_unroll(lstm, observations, state)
      logits = hk.Linear(num_actions)(embedding)
      values = jnp.squeeze(hk.Linear(1)(embedding), axis=-1)

      return (logits, values), state

    @hk.without_apply_rng
    @hk.transform
    def initial_state_fn():
      return hk.LSTM(hidden_size).initial_state(None)

    # Initial recurrent network state.
    initial_state = initial_state_fn.apply(None)

    # Make some fake data.
    observations = np.ones(shape=(sequence_len, 50))
    actions = np.random.randint(num_actions, size=sequence_len)
    rewards = np.random.rand(sequence_len)
    discounts = np.ones(shape=(sequence_len,))

    batch_tile = tree_map(lambda x: np.tile(x, [batch_size, *([1] * x.ndim)]))
    seq_tile = tree_map(lambda x: np.tile(x, [sequence_len, *([1] * x.ndim)]))

    extras = {
        'logits': np.random.rand(sequence_len, num_actions),
        'core_state': seq_tile(initial_state),
    }

    # Package up the data into a ReverbSample.
    data = adders.Step(
        observations,
        actions,
        rewards,
        discounts,
        extras=extras,
        start_of_episode=())
    data = batch_tile(data)
    sample = reverb.ReplaySample(info=None, data=data)

    # Initialise parameters.
    rng = hk.PRNGSequence(1)
    params = unroll_fn_transformed.init(next(rng), observations, initial_state)

    # Make loss function.
    loss_fn = impala.impala_loss(
        unroll_fn_transformed.apply, discount=0.99)

    # Return value should be scalar.
    loss, metrics = loss_fn(params, sample)
    loss = jax.device_get(loss)
    self.assertEqual(loss.shape, ())
    for value in metrics.values():
      value = jax.device_get(value)
      self.assertEqual(value.shape, ())


if __name__ == '__main__':
  absltest.main()
