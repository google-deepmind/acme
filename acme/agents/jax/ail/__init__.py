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

from acme.agents.jax.ail import losses, rewards
from acme.agents.jax.ail.builder import AILBuilder
from acme.agents.jax.ail.config import AILConfig
from acme.agents.jax.ail.dac import DACBuilder, DACConfig
from acme.agents.jax.ail.gail import GAILBuilder, GAILConfig
from acme.agents.jax.ail.learning import AILLearner
from acme.agents.jax.ail.networks import (
    AILNetworks,
    AIRLModule,
    DiscriminatorMLP,
    DiscriminatorModule,
    compute_ail_reward,
    make_discriminator,
)
