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

"""MPO agent module."""

from acme.agents.jax.mpo.acting import ActorState
from acme.agents.jax.mpo.acting import make_actor_core
from acme.agents.jax.mpo.builder import MPOBuilder
from acme.agents.jax.mpo.config import MPOConfig
from acme.agents.jax.mpo.learning import MPOLearner
from acme.agents.jax.mpo.networks import make_control_networks
from acme.agents.jax.mpo.networks import MPONetworks
from acme.agents.jax.mpo.types import CategoricalPolicyLossConfig
from acme.agents.jax.mpo.types import CriticType
from acme.agents.jax.mpo.types import GaussianPolicyLossConfig
from acme.agents.jax.mpo.types import PolicyLossConfig
