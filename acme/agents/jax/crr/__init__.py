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

"""Implementation of the Critic Regularized Regression (CRR) agent."""

from acme.agents.jax.crr.learning import CRRLearner
from acme.agents.jax.crr.losses import policy_loss_coeff_advantage_exp
from acme.agents.jax.crr.losses import policy_loss_coeff_advantage_indicator
from acme.agents.jax.crr.losses import policy_loss_coeff_constant
from acme.agents.jax.crr.losses import PolicyLossCoeff
from acme.agents.jax.crr.networks import CRRNetworks
from acme.agents.jax.crr.networks import make_networks
