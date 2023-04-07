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

from acme.jax.networks.atari import (
    AtariTorso,
    DeepIMPALAAtariNetwork,
    R2D2AtariNetwork,
    dqn_atari_network,
)
from acme.jax.networks.base import (
    Action,
    Entropy,
    FeedForwardNetwork,
    Logits,
    LogProb,
    LogProbFn,
    LSTMOutputs,
    NetworkOutput,
    Observation,
    Params,
    PolicyValueRNN,
    PRNGKey,
    QNetwork,
    QValues,
    RecurrentQNetwork,
    RecurrentState,
    SampleFn,
    TypedFeedForwardNetwork,
    UnrollableNetwork,
    Value,
    make_unrollable_network,
    non_stochastic_network_to_typed,
)
from acme.jax.networks.continuous import LayerNormMLP, NearZeroInitializedLinear
from acme.jax.networks.distributional import (
    CategoricalCriticHead,
    CategoricalHead,
    CategoricalValueHead,
    DiscreteValued,
    GaussianMixture,
    MultivariateNormalDiagHead,
    NormalTanhDistribution,
    TanhTransformedDistribution,
)
from acme.jax.networks.duelling import DuellingMLP
from acme.jax.networks.multiplexers import CriticMultiplexer
from acme.jax.networks.policy_value import PolicyValueHead
from acme.jax.networks.rescaling import ClipToSpec, TanhToSpec
from acme.jax.networks.resnet import DownsamplingStrategy, ResidualBlock, ResNetTorso
