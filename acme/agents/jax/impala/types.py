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

"""Some types/assumptions used in the IMPALA agent."""
from typing import Callable, Tuple

from acme.agents.jax.actor_core import RecurrentState
from acme.jax import networks
import jax.numpy as jnp


# Only simple observations & discrete action spaces for now.
Observation = jnp.ndarray
Action = int
Outputs = Tuple[Tuple[networks.Logits, networks.Value], RecurrentState]
PolicyValueInitFn = Callable[[networks.PRNGKey, Observation, RecurrentState],
                             networks.Params]
PolicyValueFn = Callable[[networks.Params, Observation, RecurrentState],
                         Outputs]
RecurrentStateInitFn = Callable[[networks.PRNGKey], networks.Params]
RecurrentStateFn = Callable[[networks.Params], RecurrentState]
