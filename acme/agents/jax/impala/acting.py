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

"""IMPALA actor implementation."""

from typing import Callable, Optional

from acme import adders
from acme import core
from acme.agents.jax.impala import types
from acme.jax import networks
from acme.jax import variable_utils
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp


class IMPALAActor(core.Actor):
  """A recurrent actor."""

  _state: hk.LSTMState
  _prev_state: hk.LSTMState
  _prev_logits: jnp.ndarray

  def __init__(
      self,
      forward_fn: networks.PolicyValueRNN,
      initial_state_fn: Callable[[], hk.LSTMState],
      rng: hk.PRNGSequence,
      variable_client: variable_utils.VariableClient,
      adder: Optional[adders.Adder] = None,
  ):

    # Store these for later use.
    self._adder = adder
    self._variable_client = variable_client
    self._forward = jax.jit(hk.transform(forward_fn).apply, backend='cpu')
    self._rng = rng

    self._params = variable_client.update_and_wait()
    self._initial_state = hk.transform(initial_state_fn).apply(None)

  def select_action(self, observation: types.Observation) -> types.Action:

    if self._state is None:
      self._state = self._initial_state

    # Forward.
    (logits, _), new_state = self._forward(self._variable_client.params,
                                           observation, self._state)

    self._prev_logits = logits
    self._prev_state = self._state
    self._state = new_state

    action = jax.random.categorical(next(self._rng), logits)

    return action

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

    # Set the state to None so that we re-initialize at the next policy call.
    self._state = None

  def observe(
      self,
      action: types.Action,
      next_timestep: dm_env.TimeStep,
  ):
    if not self._adder:
      return

    extras = {'logits': self._prev_logits, 'core_state': self._prev_state}
    self._adder.add(action, next_timestep, extras)

  def update(self):
    if self._variable_client:
      self._variable_client.update()
