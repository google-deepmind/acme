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

"""Network definitions for BVE."""

import dataclasses
from typing import Optional

from acme.jax import networks as networks_lib


@dataclasses.dataclass
class BVENetworks:
  """The network and pure functions for the BVE agent.

  Attributes:
    policy_network: The policy network.
    sample_fn: A pure function. Samples an action based on the network output.
    log_prob: A pure function. Computes log-probability for an action.
  """
  policy_network: networks_lib.TypedFeedForwardNetwork
  sample_fn: networks_lib.SampleFn
  log_prob: Optional[networks_lib.LogProbFn] = None
