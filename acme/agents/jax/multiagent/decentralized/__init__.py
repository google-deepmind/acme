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

"""Decentralized multiagent configuration."""

from acme.agents.jax.multiagent.decentralized.agents import DecentralizedMultiAgent
from acme.agents.jax.multiagent.decentralized.agents import DistributedDecentralizedMultiAgent
from acme.agents.jax.multiagent.decentralized.builder import DecentralizedMultiAgentBuilder
from acme.agents.jax.multiagent.decentralized.config import DecentralizedMultiagentConfig
from acme.agents.jax.multiagent.decentralized.factories import builder_factory
from acme.agents.jax.multiagent.decentralized.factories import default_config_factory
from acme.agents.jax.multiagent.decentralized.factories import default_logger_factory
from acme.agents.jax.multiagent.decentralized.factories import DefaultSupportedAgent
from acme.agents.jax.multiagent.decentralized.factories import network_factory
from acme.agents.jax.multiagent.decentralized.factories import policy_network_factory
