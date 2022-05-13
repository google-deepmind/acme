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

"""IMPALA networks definition."""

import dataclasses
from typing import Any, Generic, Optional, Tuple

from acme import specs
from acme.agents.jax.impala import types
from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk


@dataclasses.dataclass
class IMPALANetworks(Generic[types.RecurrentState]):

  """Pure functions representing IMPALA's recurrent network components.

  Attributes:
    forward_fn: Selects next action using the network at the given recurrent
      state.

    unroll_init_fn: Initializes params for forward_fn and unroll_fn.

    unroll_fn: Applies the unrolled network to a sequence of observations, for
      learning.
    initial_state_fn: Recurrent state at the beginning of an episode.
  """
  forward_fn: types.PolicyValueFn
  unroll_init_fn: types.PolicyValueInitFn
  unroll_fn: types.PolicyValueFn
  initial_state_fn: types.RecurrentStateFn


def make_haiku_networks(
    env_spec: specs.EnvironmentSpec,
    forward_fn: Any,
    initial_state_fn: Any,
    unroll_fn: Any) -> IMPALANetworks[types.RecurrentState]:
  """Builds functional impala network from recurrent model definitions."""
  # Make networks purely functional.
  forward_hk = hk.without_apply_rng(hk.transform(forward_fn))
  initial_state_hk = hk.without_apply_rng(hk.transform(initial_state_fn))
  unroll_hk = hk.without_apply_rng(hk.transform(unroll_fn))

  # Note: batch axis is not needed for the actors.
  dummy_obs = utils.zeros_like(env_spec.observations)
  dummy_obs_sequence = utils.add_batch_dim(dummy_obs)
  def unroll_init_fn(
      rng: networks_lib.PRNGKey,
      initial_state: types.RecurrentState) -> hk.Params:
    return unroll_hk.init(rng, dummy_obs_sequence, initial_state)

  return IMPALANetworks(
      forward_fn=forward_hk.apply,
      unroll_init_fn=unroll_init_fn,
      unroll_fn=unroll_hk.apply,
      initial_state_fn=(
          lambda rng: initial_state_hk.apply(initial_state_hk.init(rng))))


HaikuLSTMOutputs = Tuple[Tuple[networks_lib.Logits, networks_lib.Value],
                         hk.LSTMState]


def make_atari_networks(env_spec: specs.EnvironmentSpec
                        ) -> IMPALANetworks[hk.LSTMState]:
  """Builds default IMPALA networks for Atari games."""

  def forward_fn(inputs: types.Observation, state: hk.LSTMState
                 ) -> HaikuLSTMOutputs:
    model = networks_lib.DeepIMPALAAtariNetwork(env_spec.actions.num_values)
    return model(inputs, state)

  def initial_state_fn(batch_size: Optional[int] = None) -> hk.LSTMState:
    model = networks_lib.DeepIMPALAAtariNetwork(env_spec.actions.num_values)
    return model.initial_state(batch_size)

  def unroll_fn(inputs: types.Observation, state: hk.LSTMState
                ) -> HaikuLSTMOutputs:
    model = networks_lib.DeepIMPALAAtariNetwork(env_spec.actions.num_values)
    return model.unroll(inputs, state)

  return make_haiku_networks(
      env_spec=env_spec, forward_fn=forward_fn,
      initial_state_fn=initial_state_fn, unroll_fn=unroll_fn)
