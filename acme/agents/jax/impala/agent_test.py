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

"""Tests for IMPALA agent."""

from typing import Optional

from absl.testing import absltest

import acme
from acme import specs
from acme.agents.jax import impala
from acme.jax import networks as networks_lib
from acme.testing import fakes

import haiku as hk
import jax.numpy as jnp
import numpy as np


class MyNetwork(hk.RNNCore):
  """A simple recurrent network for testing."""

  def __init__(self, num_actions: int):
    super().__init__(name='my_network')
    self._torso = hk.Sequential([
        lambda x: jnp.reshape(x, [np.prod(x.shape)]),
        hk.nets.MLP([50, 50]),
    ])
    self._core = hk.LSTM(20)
    self._head = networks_lib.PolicyValueHead(num_actions)

  def __call__(self, inputs, state):
    embeddings = self._torso(inputs)
    embeddings, new_state = self._core(embeddings, state)
    logits, value = self._head(embeddings)
    return (logits, value), new_state

  def initial_state(self, batch_size: int):
    return self._core.initial_state(batch_size)


class IMPALATest(absltest.TestCase):

  def test_impala(self):
    # Create a fake environment to test with.
    environment = fakes.DiscreteEnvironment(
        num_actions=5,
        num_observations=10,
        obs_shape=(10, 5),
        obs_dtype=np.float32,
        episode_length=10)
    spec = specs.make_environment_spec(environment)

    def forward_fn(x, s):
      model = MyNetwork(spec.actions.num_values)
      return model(x, s)

    def initial_state_fn(batch_size: Optional[int] = None):
      model = MyNetwork(spec.actions.num_values)
      return model.initial_state(batch_size)

    def unroll_fn(inputs, state):
      model = MyNetwork(spec.actions.num_values)
      return hk.static_unroll(model, inputs, state)

    # We pass pure, Haiku-agnostic functions to the agent.
    networks = impala.make_haiku_networks(
        env_spec=spec,
        forward_fn=forward_fn, initial_state_fn=initial_state_fn,
        unroll_fn=unroll_fn)

    # Construct the agent.
    config = impala.IMPALAConfig(
        sequence_length=3,
        sequence_period=3,
        batch_size=6,
    )
    agent = impala.IMPALAFromConfig(
        environment_spec=spec,
        forward_fn=networks.forward_fn,
        unroll_init_fn=networks.unroll_init_fn,
        unroll_fn=networks.unroll_fn,
        initial_state_init_fn=networks.initial_state_init_fn,
        initial_state_fn=networks.initial_state_fn,
        config=config,
    )

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent)
    loop.run(num_episodes=20)


if __name__ == '__main__':
  absltest.main()
