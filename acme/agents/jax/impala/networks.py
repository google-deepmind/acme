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

"""IMPALA networks definition."""

import dataclasses

from acme.agents.jax.impala import types


@dataclasses.dataclass
class IMPALANetworks:
  """Pure functions representing IMPALA's recurrent network components.

  Attributes:
    forward_fn: Selects next action using the network at the given recurrent
      state.
    unroll_init_fn: Initializes params for unroll_fn.
    unroll_fn: Applies the unrolled network to a sequence of observations, for
      learning.
    initial_state_init_fn: Initializes params for initial_state_fn.
    initial_state_fn: Recurrent state at the beginning of an episode.
  """
  forward_fn: types.PolicyValueFn
  unroll_init_fn: types.PolicyValueInitFn
  unroll_fn: types.PolicyValueFn
  initial_state_init_fn: types.RecurrentStateInitFn
  initial_state_fn: types.RecurrentStateFn
