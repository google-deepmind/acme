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

"""Defines the D4PG agent class, using JAX."""

from typing import Callable, Optional

from acme import specs
from acme.agents.jax.d4pg import builder as d4pg_builder
from acme.agents.jax.d4pg import config as d4pg_config
from acme.agents.jax.d4pg import networks as d4pg_networks
from acme.jax.layouts import local_layout
from acme.utils import counting

NetworkFactory = Callable[[specs.EnvironmentSpec], d4pg_networks.D4PGNetworks]


class D4PG(local_layout.LocalLayout):
  """Local agent for D4PG."""

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      network: d4pg_networks.D4PGNetworks,
      config: d4pg_config.D4PGConfig,
      random_seed: int,
      counter: Optional[counting.Counter] = None,
  ):
    self.builder = d4pg_builder.D4PGBuilder(config)
    super().__init__(
        seed=random_seed,
        environment_spec=spec,
        builder=self.builder,
        networks=network,
        policy_network=d4pg_networks.get_default_behavior_policy(
            network, config),
        batch_size=config.batch_size,
        num_sgd_steps_per_step=config.num_sgd_steps_per_step,
        counter=counter,
    )
