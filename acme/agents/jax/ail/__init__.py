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

"""Implementations of a AIL agent."""

from acme.agents.jax.ail import losses
from acme.agents.jax.ail import rewards
from acme.agents.jax.ail.agents import AIL
from acme.agents.jax.ail.agents import DistributedAIL
from acme.agents.jax.ail.builder import AILBuilder
from acme.agents.jax.ail.config import AILConfig
from acme.agents.jax.ail.dac_agents import DAC
from acme.agents.jax.ail.dac_agents import DACConfig
from acme.agents.jax.ail.dac_agents import DistributedDAC
from acme.agents.jax.ail.gail_agents import DistributedGAIL
from acme.agents.jax.ail.gail_agents import GAIL
from acme.agents.jax.ail.gail_agents import GAILConfig
from acme.agents.jax.ail.learning import AILLearner
from acme.agents.jax.ail.networks import AILNetworks
from acme.agents.jax.ail.networks import AIRLModule
from acme.agents.jax.ail.networks import compute_ail_reward
from acme.agents.jax.ail.networks import DiscriminatorMLP
from acme.agents.jax.ail.networks import DiscriminatorModule
from acme.agents.jax.ail.networks import make_discriminator
