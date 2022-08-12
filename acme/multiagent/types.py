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

"""Types for multiagent setups."""

from typing import Any, Callable, Dict, Tuple

from acme import specs
from acme.agents.jax import builders as jax_builders
from acme.utils.loggers import base
import reverb


# Sub-agent types
AgentID = str
EvalMode = bool
GenericAgent = Any
AgentConfig = Any
Networks = Any
PolicyNetwork = Any
LoggerFn = Callable[[], base.Logger]
InitNetworkFn = Callable[[GenericAgent, specs.EnvironmentSpec], Networks]
InitPolicyNetworkFn = Callable[
    [GenericAgent, Networks, specs.EnvironmentSpec, AgentConfig, bool],
    Networks]
InitBuilderFn = Callable[[GenericAgent, AgentConfig],
                         jax_builders.GenericActorLearnerBuilder]

# Multiagent types
MultiAgentLoggerFn = Dict[AgentID, LoggerFn]
MultiAgentNetworks = Dict[AgentID, Networks]
MultiAgentPolicyNetworks = Dict[AgentID, PolicyNetwork]
MultiAgentSample = Tuple[reverb.ReplaySample, ...]
NetworkFactory = Callable[[specs.EnvironmentSpec], MultiAgentNetworks]
PolicyFactory = Callable[[MultiAgentNetworks, EvalMode],
                         MultiAgentPolicyNetworks]
BuilderFactory = Callable[[
    Dict[AgentID, GenericAgent],
    Dict[AgentID, AgentConfig],
], Dict[AgentID, jax_builders.GenericActorLearnerBuilder]]
