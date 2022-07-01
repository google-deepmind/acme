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

"""Implementation of an R2D2 agent."""

from acme.agents.jax.r2d2.actor import EpsilonRecurrentPolicy
from acme.agents.jax.r2d2.actor import make_behavior_policy
from acme.agents.jax.r2d2.agents import DistributedR2D2
from acme.agents.jax.r2d2.agents import DistributedR2D2FromConfig
from acme.agents.jax.r2d2.agents import R2D2
from acme.agents.jax.r2d2.builder import R2D2Builder
from acme.agents.jax.r2d2.config import R2D2Config
from acme.agents.jax.r2d2.learning import R2D2Learner
from acme.agents.jax.r2d2.networks import make_atari_networks
from acme.agents.jax.r2d2.networks import make_networks
from acme.agents.jax.r2d2.networks import R2D2Networks
