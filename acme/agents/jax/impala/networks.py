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

from typing import Optional, Tuple

from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.jax import utils
import haiku as hk


IMPALANetworks = networks_lib.UnrollableNetwork
NetworkOutput = networks_lib.NetworkOutput
RecurrentState = networks_lib.RecurrentState


def make_atari_networks(env_spec: specs.EnvironmentSpec) -> IMPALANetworks:
  """Builds default IMPALA networks for Atari games."""

  dummy_observation = utils.zeros_like(env_spec.observations)

  def make_unrollable_network_functions():
    model = networks_lib.DeepIMPALAAtariNetwork(env_spec.actions.num_values)
    apply = model.__call__

    def init() -> Tuple[NetworkOutput, RecurrentState]:
      return model(dummy_observation, model.initial_state(None))

    return init, (apply, model.unroll, model.initial_state)

  # Transform and unpack pure functions
  f = hk.multi_transform(make_unrollable_network_functions)
  apply, unroll, initial_state_fn = f.apply

  def init_initial_state(key: jax_types.PRNGKey,
                         batch_size: Optional[int]) -> RecurrentState:
    # TODO(b/244311990): Consider supporting parameterized and learnable initial
    # state functions.
    no_params = None
    return initial_state_fn(no_params, key, batch_size)

  return IMPALANetworks(f.init, apply, unroll, init_initial_state)
