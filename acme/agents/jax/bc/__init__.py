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

"""Implementation of a behavior cloning (BC) agent."""

from acme.agents.jax.bc import pretraining
from acme.agents.jax.bc.builder import BCBuilder
from acme.agents.jax.bc.config import BCConfig
from acme.agents.jax.bc.learning import BCLearner
from acme.agents.jax.bc.losses import BCLoss
from acme.agents.jax.bc.losses import logp
from acme.agents.jax.bc.losses import mse
from acme.agents.jax.bc.losses import peerbc
from acme.agents.jax.bc.losses import rcal
from acme.agents.jax.bc.networks import BCNetworks
from acme.agents.jax.bc.networks import BCPolicyNetwork
from acme.agents.jax.bc.networks import convert_policy_value_to_bc_network
from acme.agents.jax.bc.networks import convert_to_bc_network
