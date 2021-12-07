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

"""Tests for R2D2 agent."""

from typing import Optional

from absl.testing import absltest
import acme
from acme import specs
from acme.agents.jax import r2d2
from acme.jax import networks as networks_lib
from acme.testing import fakes
from acme.utils import counting
import haiku as hk
import jax
import numpy as np


class RDQN(hk.RNNCore):
  """A simple recurrent network for testing."""

  def __init__(self, num_actions: int):
    super().__init__(name='my_network')
    self._torso = hk.Sequential([
        hk.Flatten(),
        hk.nets.MLP([50, 50]),
    ])
    self._core = hk.LSTM(20)
    self._head = networks_lib.PolicyValueHead(num_actions)

  def __call__(self, inputs, state):
    embeddings = self._torso(inputs)
    embeddings, new_state = self._core(embeddings, state)
    logits, _ = self._head(embeddings)
    return logits, new_state

  def initial_state(self, batch_size: int):
    return self._core.initial_state(batch_size)

  def unroll(self, inputs, state):
    embeddings = jax.vmap(self._torso)(inputs)  # [T D]
    embeddings, new_states = hk.static_unroll(self._core, embeddings, state)
    logits, _ = self._head(embeddings)
    return logits, new_states


def make_networks(spec, batch_size):
  """Creates networks used by the agent."""

  def forward_fn(inputs, state):
    model = RDQN(spec.actions.num_values)
    return model(inputs, state)

  def initial_state_fn(batch_size: Optional[int] = None):
    model = RDQN(spec.actions.num_values)
    return model.initial_state(batch_size)

  def unroll_fn(inputs, state):
    model = RDQN(spec.actions.num_values)
    return model.unroll(inputs, state)

  return r2d2.make_networks(
      forward_fn=forward_fn,
      initial_state_fn=initial_state_fn,
      unroll_fn=unroll_fn,
      env_spec=spec,
      batch_size=batch_size)


class R2D2Test(absltest.TestCase):

  def test_r2d2(self):
    # Create a fake environment to test with.
    environment = fakes.DiscreteEnvironment(
        num_actions=5,
        num_observations=10,
        obs_shape=(10, 5),
        obs_dtype=np.float32,
        episode_length=10)
    spec = specs.make_environment_spec(environment)

    config = r2d2.R2D2Config(
        batch_size=1,
        trace_length=5,
        sequence_period=1,
        samples_per_insert=0.,
        min_replay_size=1,
        burn_in_length=1)

    counter = counting.Counter()
    agent = r2d2.R2D2(
        spec=spec,
        networks=make_networks(spec, config.batch_size),
        config=config,
        seed=0,
        counter=counter,
    )

    # Try running the environment loop. We have no assertions here because all
    # we care about is that the agent runs without raising any errors.
    loop = acme.EnvironmentLoop(environment, agent, counter=counter)
    loop.run(num_episodes=20)


if __name__ == '__main__':
  absltest.main()
