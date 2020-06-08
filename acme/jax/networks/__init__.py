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

from acme.jax.networks.atari import DeepIMPALAAtariNetwork
from acme.jax.networks.atari import dqn_atari_network
from acme.jax.networks.base import PolicyValueRNN
from acme.jax.networks.base import QNetwork
from acme.jax.networks.continuous import LayerNormMLP
from acme.jax.networks.continuous import NearZeroInitializedLinear
from acme.jax.networks.policy_value import PolicyValueHead
from acme.jax.networks.rescaling import ClipToSpec
from acme.jax.networks.rescaling import TanhToSpec
