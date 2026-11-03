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

"""Implicit Q-Learning (IQL) agent implementation."""

from acme.agents.jax.iql.builder import IQLBuilder
from acme.agents.jax.iql.config import IQLConfig
from acme.agents.jax.iql.learning import IQLLearner
from acme.agents.jax.iql.networks import IQLNetworks
from acme.agents.jax.iql.networks import make_networks

__all__ = [
    'IQLBuilder',
    'IQLConfig',
    'IQLLearner',
    'IQLNetworks',
    'make_networks',
]
