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

"""Simple JAX actors."""

from typing import Callable, Optional

from acme import adders
from acme import core
from acme import types
from acme.jax import utils
from acme.jax import variable_utils

import dm_env
import haiku as hk
import jax
import jax.numpy as jnp

# Useful type aliases.
RNGKey = jnp.ndarray
Observation = types.NestedArray
Action = types.NestedArray

# Signature for a function that samples from a parameterised stochastic policy.
FeedForwardPolicy = Callable[[hk.Params, RNGKey, Observation], Action]


class FeedForwardActor(core.Actor):
  """A simple feed-forward actor implemented in JAX."""

  def __init__(
      self,
      policy: FeedForwardPolicy,
      rng: hk.PRNGSequence,
      variable_client: variable_utils.VariableClient,
      adder: Optional[adders.Adder] = None,
  ):
    self._rng = rng
    self._policy = jax.jit(policy, backend='cpu')

    self._adder = adder
    self._client = variable_client

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    key = next(self._rng)
    observation = utils.add_batch_dim(observation)
    action = self._policy(self._client.params, key, observation)
    return utils.to_numpy_squeeze(action)

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add(action, next_timestep)

  def update(self):
    self._client.update()
