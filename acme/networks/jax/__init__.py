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

"""JAX networks implemented with Haiku."""

from acme.networks.jax.atari import DeepIMPALAAtariNetwork
from acme.networks.jax.atari import dqn_atari_network
from acme.networks.jax.base import PolicyValueRNN
from acme.networks.jax.base import QNetwork
from acme.networks.jax.base import RNNState
from acme.networks.jax.continuous import LayerNormMLP
from acme.networks.jax.continuous import NearZeroInitializedLinear
from acme.networks.jax.policy_value import PolicyValueHead
from acme.networks.jax.rescaling import ClipToSpec
from acme.networks.jax.rescaling import TanhToSpec
