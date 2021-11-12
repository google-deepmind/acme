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

"""Implementations of a D4PG agent."""

from acme.agents.jax.d4pg.agents import D4PG
from acme.agents.jax.d4pg.agents import DistributedD4PG
from acme.agents.jax.d4pg.builder import D4PGBuilder
from acme.agents.jax.d4pg.builder import D4PGConfig
from acme.agents.jax.d4pg.builder import D4PGNetworks
from acme.agents.jax.d4pg.builder import get_default_behavior_policy
from acme.agents.jax.d4pg.builder import get_default_eval_policy
from acme.agents.jax.d4pg.learning import D4PGLearner
