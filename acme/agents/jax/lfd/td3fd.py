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

"""TD3 agent learning from demonstrations."""

import dataclasses
from typing import Callable, Iterator

from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import td3
from acme.agents.jax.lfd import builder
from acme.agents.jax.lfd import config
import reverb


@dataclasses.dataclass
class TD3fDConfig:
  """Configuration options specific to TD3 with demonstrations.

  Attributes:
    lfd_config: LfD config.
    td3_config: TD3 config.
  """
  lfd_config: config.LfdConfig
  td3_config: td3.TD3Config


class TD3fDBuilder(builder.LfdBuilder[td3.TD3Networks,
                                      actor_core_lib.FeedForwardPolicy,
                                      reverb.ReplaySample]):
  """Builder for TD3 agent learning from demonstrations."""

  def __init__(self, td3_fd_config: TD3fDConfig,
               lfd_iterator_fn: Callable[[], Iterator[builder.LfdStep]]):
    td3_builder = td3.TD3Builder(td3_fd_config.td3_config)
    super().__init__(td3_builder, lfd_iterator_fn, td3_fd_config.lfd_config)
