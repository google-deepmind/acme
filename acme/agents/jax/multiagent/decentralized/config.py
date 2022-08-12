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

"""Decentralized multiagent config."""

import dataclasses
from typing import Dict

from acme.multiagent import types


@dataclasses.dataclass
class DecentralizedMultiagentConfig:
  """Configuration options for decentralized multiagent."""
  sub_agent_configs: Dict[types.AgentID, types.AgentConfig]
  batch_size: int = 256
  prefetch_size: int = 2
