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

"""Defines distributed and local TD3 agents, using JAX."""

from typing import Callable, Optional

from acme import specs
from acme.agents.jax.td3 import builder
from acme.agents.jax.td3 import config as td3_config
from acme.agents.jax.td3 import networks
from acme.jax.layouts import local_layout
from acme.utils import counting

NetworkFactory = Callable[[specs.EnvironmentSpec], networks.TD3Networks]


class TD3(local_layout.LocalLayout):
  """Local agent for TD3."""

  def __init__(
      self,
      spec: specs.EnvironmentSpec,
      network: networks.TD3Networks,
      config: td3_config.TD3Config,
      seed: int,
      counter: Optional[counting.Counter] = None,
  ):
    behavior_policy = networks.get_default_behavior_policy(
        networks=network, action_specs=spec.actions, sigma=config.sigma)

    self.builder = builder.TD3Builder(config)
    super().__init__(
        seed=seed,
        environment_spec=spec,
        builder=self.builder,
        networks=network,
        policy_network=behavior_policy,
        batch_size=config.batch_size,
        prefetch_size=config.prefetch_size,
        num_sgd_steps_per_step=config.num_sgd_steps_per_step,
        counter=counter,
    )
