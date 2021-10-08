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

"""TD3 agent."""

from acme.agents.jax.td3.agents import DistributedTD3
from acme.agents.jax.td3.agents import TD3
from acme.agents.jax.td3.builder import TD3Builder
from acme.agents.jax.td3.config import TD3Config
from acme.agents.jax.td3.learning import TD3Learner
from acme.agents.jax.td3.networks import get_default_behavior_policy
from acme.agents.jax.td3.networks import make_networks
from acme.agents.jax.td3.networks import TD3Networks
