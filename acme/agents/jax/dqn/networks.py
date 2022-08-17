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

"""Network definitions for DQN."""

import dataclasses
from typing import Callable, Optional

from acme.jax import networks as networks_lib
from acme.jax import types
import rlax

Epsilon = float
EpsilonPolicy = Callable[[
    networks_lib.Params, networks_lib.PRNGKey, networks_lib.Observation, Epsilon
], networks_lib.Action]
EpsilonSampleFn = Callable[[networks_lib.NetworkOutput, types.PRNGKey, Epsilon],
                           networks_lib.Action]
EpsilonLogProbFn = Callable[
    [networks_lib.NetworkOutput, networks_lib.Action, Epsilon],
    networks_lib.LogProb]


def default_sample_fn(action_values: networks_lib.NetworkOutput,
                      key: types.PRNGKey,
                      epsilon: Epsilon) -> networks_lib.Action:
  return rlax.epsilon_greedy(epsilon).sample(key, action_values)


@dataclasses.dataclass
class DQNNetworks:
  """The network and pure functions for the DQN agent.

  Attributes:
    policy_network: The policy network.
    sample_fn: A pure function. Samples an action based on the network output.
    log_prob: A pure function. Computes log-probability for an action.
  """
  policy_network: networks_lib.TypedFeedForwardNetwork
  sample_fn: EpsilonSampleFn = default_sample_fn
  log_prob: Optional[EpsilonLogProbFn] = None
