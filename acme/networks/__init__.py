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

"""Useful network definitions."""

from acme.networks.atari import AtariTorso
from acme.networks.atari import DeepIMPALAAtariNetwork
from acme.networks.atari import DQNAtariNetwork
from acme.networks.atari import IMPALAAtariNetwork
from acme.networks.atari import R2D2AtariNetwork
from acme.networks.base import RNNCore
from acme.networks.continuous import LayerNormAndResidualMLP
from acme.networks.continuous import LayerNormMLP
from acme.networks.continuous import NearZeroInitializedLinear
from acme.networks.distributional import ApproximateMode
from acme.networks.distributional import DiscreteValuedDistribution
from acme.networks.distributional import DiscreteValuedHead
from acme.networks.distributional import GaussianMixtureHead
from acme.networks.distributional import MultivariateNormalDiagHead
from acme.networks.duelling import DuellingMLP
from acme.networks.multiplexers import CriticMultiplexer
from acme.networks.noise import ClippedGaussian
from acme.networks.policy_value import PolicyValueHead
from acme.networks.rescaling import ClipToSpec
from acme.networks.rescaling import RescaleToSpec
from acme.networks.rescaling import TanhToSpec
from acme.networks.stochastic import ExpQWeightedPolicy
from acme.networks.stochastic import StochasticMeanHead
from acme.networks.stochastic import StochasticModeHead
from acme.networks.stochastic import StochasticSamplingHead
from acme.networks.vision import ResNetTorso
