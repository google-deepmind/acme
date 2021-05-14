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
from acme.agents.jax.impala import agent as impala_agent
from acme.jax import networks
from acme.testing import fakes

import haiku as hk
import jax.numpy as jnp
import numpy as np


class MyNetwork(hk.RNNCore):
  """A simple recurrent network for testing."""

  def __init__(self, num_actions: int):
    super().__init__(name="my_network")
    self._torso = hk.Sequential([
        hk.Flatten(),
        hk.nets.MLP([50, 50]),
    ])
    self._core = hk.LSTM(20)
    self._head = networks.PolicyValueHead(num_actions)

  def __call__(self, inputs, state):
    embeddings = self.embed(inputs)
    embeddings, new_state = self._core(embeddings, state)
    logits, value = self._head(embeddings)
    return (logits, value), new_state

  def initial_state(self, batch_size: int):
    return self._core.initial_state(batch_size)

  def embed(self, observation):
    if observation.ndim not in [2, 3]:
      raise ValueError("Expects inputs to have rank 3 (unbatched) or 4 (batched), "
                       f"got {observation.ndim} instead")
    expand_obs = observation.ndim == 2
    if expand_obs:
      observation = jnp.expand_dims(observation, 0)
    features = self._torso(observation.astype(jnp.float32))
    if expand_obs:
      features = jnp.squeeze(features, 0)
    return features

  def unroll(self, inputs, initial_state, start_of_episode=None):
    embeddings = self.embed(inputs)
    core = self._core
    if start_of_episode is not None:
      embeddings = (embeddings, start_of_episode)
      core = hk.ResetCore(self._core)
    initial_state = hk.LSTMState(initial_state.hidden, initial_state.cell)
    core_outputs, final_state = hk.static_unroll(core, embeddings, initial_state)
    return self._head(core_outputs), final_state


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

    def unroll_fn(inputs, state, start_of_episode=None):
      model = MyNetwork(spec.actions.num_values)
      return model.unroll(inputs, state, start_of_episode)

    # We pass pure, Haiku-agnostic functions to the agent.
    forward_fn_transformed = hk.without_apply_rng(hk.transform(
        forward_fn,
        apply_rng=True))
    unroll_fn_transformed = hk.without_apply_rng(hk.transform(
        unroll_fn,
        apply_rng=True))
    initial_state_fn_transformed = hk.without_apply_rng(hk.transform(
        initial_state_fn,
        apply_rng=True))

    # Construct the agent.
    config = impala_agent.IMPALAConfig(
        sequence_length=3,
        sequence_period=3,
        batch_size=6,
        break_end_of_episode=True,
    )
    agent = impala.IMPALAFromConfig(
        environment_spec=spec,
        forward_fn=forward_fn_transformed.apply,
        initial_state_init_fn=initial_state_fn_transformed.init,
        initial_state_fn=initial_state_fn_transformed.apply,
        unroll_init_fn=unroll_fn_transformed.init,
        unroll_fn=unroll_fn_transformed.apply,
        config=config,
    )

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent)
    loop.run(num_episodes=20)


if __name__ == '__main__':
  absltest.main()
