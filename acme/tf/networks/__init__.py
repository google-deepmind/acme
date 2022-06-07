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

from acme.tf.networks.atari import AtariTorso
from acme.tf.networks.atari import DeepIMPALAAtariNetwork
from acme.tf.networks.atari import DQNAtariNetwork
from acme.tf.networks.atari import IMPALAAtariNetwork
from acme.tf.networks.atari import R2D2AtariNetwork
from acme.tf.networks.base import DistributionalModule
from acme.tf.networks.base import Module
from acme.tf.networks.base import RNNCore
from acme.tf.networks.continuous import LayerNormAndResidualMLP
from acme.tf.networks.continuous import LayerNormMLP
from acme.tf.networks.continuous import NearZeroInitializedLinear
from acme.tf.networks.discrete import DiscreteFilteredQNetwork
from acme.tf.networks.distributional import ApproximateMode
from acme.tf.networks.distributional import DiscreteValuedHead
from acme.tf.networks.distributional import MultivariateGaussianMixture
from acme.tf.networks.distributional import MultivariateNormalDiagHead
from acme.tf.networks.distributional import UnivariateGaussianMixture
from acme.tf.networks.distributions import DiscreteValuedDistribution
from acme.tf.networks.duelling import DuellingMLP
from acme.tf.networks.masked_epsilon_greedy import NetworkWithMaskedEpsilonGreedy
from acme.tf.networks.multihead import Multihead
from acme.tf.networks.multiplexers import CriticMultiplexer
from acme.tf.networks.noise import ClippedGaussian
from acme.tf.networks.policy_value import PolicyValueHead
from acme.tf.networks.recurrence import CriticDeepRNN
from acme.tf.networks.recurrence import DeepRNN
from acme.tf.networks.recurrence import LSTM
from acme.tf.networks.recurrence import RecurrentExpQWeightedPolicy
from acme.tf.networks.rescaling import ClipToSpec
from acme.tf.networks.rescaling import RescaleToSpec
from acme.tf.networks.rescaling import TanhToSpec
from acme.tf.networks.stochastic import ExpQWeightedPolicy
from acme.tf.networks.stochastic import StochasticMeanHead
from acme.tf.networks.stochastic import StochasticModeHead
from acme.tf.networks.stochastic import StochasticSamplingHead
from acme.tf.networks.vision import DrQTorso
from acme.tf.networks.vision import ResNetTorso

# For backwards compatibility.
GaussianMixtureHead = UnivariateGaussianMixture

try:
  # pylint: disable=g-bad-import-order,g-import-not-at-top
  from acme.tf.networks.legal_actions import MaskedSequential
  from acme.tf.networks.legal_actions import EpsilonGreedy
except ImportError:
  pass

# Internal imports.
from acme.tf.networks.quantile import IQNNetwork
