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

"""R2D2 Networks."""

import dataclasses
from typing import Any, Optional

from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.jax import utils
import haiku as hk
import jax


@dataclasses.dataclass
class R2D2Networks:
  """Network and pure functions for the R2D2 agent.."""
  forward: networks_lib.FeedForwardNetwork
  unroll: networks_lib.FeedForwardNetwork
  initial_state: networks_lib.FeedForwardNetwork


def make_networks(env_spec: specs.EnvironmentSpec, forward_fn: Any,
                  initial_state_fn: Any, unroll_fn: Any,
                  batch_size: int) -> R2D2Networks:
  """Builds functional r2d2 network from recurrent model definitions."""
  del batch_size

  # Make networks purely functional.
  forward_hk = hk.transform(forward_fn)
  initial_state_hk = hk.transform(initial_state_fn)
  unroll_hk = hk.transform(unroll_fn)

  # Define networks init functions.
  def unroll_init_fn(rng: jax_types.PRNGKey,
                     initial_state: hk.LSTMState) -> hk.Params:
    del initial_state
    init_state_params_rng, init_state_rng, unroll_rng = jax.random.split(rng, 3)
    init_state_params = initial_state_hk.init(init_state_params_rng)
    dummy_initial_state = initial_state_hk.apply(init_state_params,
                                                 init_state_rng, 1)
    dummy_obs = utils.zeros_like(env_spec.observations)
    for _ in ('batch', 'time'):  # Add time and batch dimensions.
      dummy_obs = utils.add_batch_dim(dummy_obs)
    return unroll_hk.init(unroll_rng, dummy_obs, dummy_initial_state)

  # Make FeedForwardNetworks.
  forward = networks_lib.FeedForwardNetwork(
      init=forward_hk.init, apply=forward_hk.apply)
  unroll = networks_lib.FeedForwardNetwork(
      init=unroll_init_fn, apply=unroll_hk.apply)
  initial_state = networks_lib.FeedForwardNetwork(*initial_state_hk)
  return R2D2Networks(
      forward=forward, unroll=unroll, initial_state=initial_state)


def make_atari_networks(batch_size: int,
                        env_spec: specs.EnvironmentSpec) -> R2D2Networks:
  """Builds default R2D2 networks for Atari games."""

  def make_model() -> networks_lib.R2D2AtariNetwork:
    return networks_lib.R2D2AtariNetwork(env_spec.actions.num_values)

  def forward_fn(x, s):
    return make_model()(x, s)

  def initial_state_fn(batch_size: Optional[int] = None):
    return make_model().initial_state(batch_size)

  def unroll_fn(inputs, state):
    return make_model().unroll(inputs, state)

  return make_networks(
      env_spec=env_spec,
      forward_fn=forward_fn,
      initial_state_fn=initial_state_fn,
      unroll_fn=unroll_fn,
      batch_size=batch_size)
