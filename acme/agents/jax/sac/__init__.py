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

"""SAC agent."""

from acme.agents.jax.sac.agents import DistributedSAC
from acme.agents.jax.sac.agents import SAC
from acme.agents.jax.sac.builder import SACBuilder
from acme.agents.jax.sac.config import SACConfig
from acme.agents.jax.sac.config import target_entropy_from_env_spec
from acme.agents.jax.sac.learning import SACLearner
from acme.agents.jax.sac.networks import apply_policy_and_sample
from acme.agents.jax.sac.networks import make_networks
from acme.agents.jax.sac.networks import SACNetworks
