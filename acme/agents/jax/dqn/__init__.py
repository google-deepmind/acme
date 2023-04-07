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

"""Implementation of a deep Q-networks (DQN) agent."""

from acme.agents.jax.dqn.actor import (
    DQNPolicy,
    Epsilon,
    EpsilonPolicy,
    behavior_policy,
    default_behavior_policy,
)
from acme.agents.jax.dqn.builder import DistributionalDQNBuilder, DQNBuilder
from acme.agents.jax.dqn.config import DQNConfig
from acme.agents.jax.dqn.learning import DQNLearner
from acme.agents.jax.dqn.learning_lib import LossExtra, LossFn, ReverbUpdate, SGDLearner
from acme.agents.jax.dqn.losses import (
    PrioritizedCategoricalDoubleQLearning,
    PrioritizedDoubleQLearning,
    QLearning,
    QrDqn,
)
from acme.agents.jax.dqn.networks import DQNNetworks
