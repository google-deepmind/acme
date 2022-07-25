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

"""SAC agent learning from demonstrations."""

import dataclasses
from typing import Callable, Iterator

from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import sac
from acme.agents.jax.lfd import builder
from acme.agents.jax.lfd import config
import reverb


@dataclasses.dataclass
class SACfDConfig:
  """Configuration options specific to SAC with demonstrations.

  Attributes:
    lfd_config: LfD config.
    sac_config: SAC config.
  """
  lfd_config: config.LfdConfig
  sac_config: sac.SACConfig


class SACfDBuilder(builder.LfdBuilder[sac.SACNetworks,
                                      actor_core_lib.FeedForwardPolicy,
                                      reverb.ReplaySample]):
  """Builder for SAC agent learning from demonstrations."""

  def __init__(self, sac_fd_config: SACfDConfig,
               lfd_iterator_fn: Callable[[], Iterator[builder.LfdStep]]):
    sac_builder = sac.SACBuilder(sac_fd_config.sac_config)
    super().__init__(sac_builder, lfd_iterator_fn, sac_fd_config.lfd_config)
