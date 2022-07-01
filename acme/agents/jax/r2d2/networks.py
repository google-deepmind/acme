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
from acme.jax import utils
import haiku as hk


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

  # Make networks purely functional.
  forward_hk = hk.transform(forward_fn)
  initial_state_hk = hk.transform(initial_state_fn)
  unroll_hk = hk.transform(unroll_fn)

  # Define networks init functions.
  def initial_state_init_fn(rng, batch_size):
    return initial_state_hk.init(rng, batch_size)
  dummy_obs_batch = utils.tile_nested(
      utils.zeros_like(env_spec.observations), batch_size)
  dummy_obs_sequence = utils.add_batch_dim(dummy_obs_batch)
  def unroll_init_fn(rng, initial_state):
    return unroll_hk.init(rng, dummy_obs_sequence, initial_state)

  # Make FeedForwardNetworks.
  forward = networks_lib.FeedForwardNetwork(
      init=forward_hk.init, apply=forward_hk.apply)
  unroll = networks_lib.FeedForwardNetwork(
      init=unroll_init_fn, apply=unroll_hk.apply)
  initial_state = networks_lib.FeedForwardNetwork(
      init=initial_state_init_fn, apply=initial_state_hk.apply)
  return R2D2Networks(
      forward=forward, unroll=unroll, initial_state=initial_state)


def make_atari_networks(batch_size, env_spec):
  """Builds default R2D2 networks for Atari games."""

  def forward_fn(x, s):
    model = networks_lib.R2D2AtariNetwork(env_spec.actions.num_values)
    return model(x, s)

  def initial_state_fn(batch_size: Optional[int] = None):
    model = networks_lib.R2D2AtariNetwork(env_spec.actions.num_values)
    return model.initial_state(batch_size)

  def unroll_fn(inputs, state):
    model = networks_lib.R2D2AtariNetwork(env_spec.actions.num_values)
    return model.unroll(inputs, state)

  return make_networks(env_spec=env_spec, forward_fn=forward_fn,
                       initial_state_fn=initial_state_fn, unroll_fn=unroll_fn,
                       batch_size=batch_size)
