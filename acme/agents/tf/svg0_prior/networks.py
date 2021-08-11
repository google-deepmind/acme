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

"""Shared helpers for different experiment flavours."""

import functools
from typing import Mapping, Sequence, Optional

from acme import specs
from acme import types
from acme.agents.tf.svg0_prior import utils as svg0_utils
from acme.tf import networks
from acme.tf import utils as tf2_utils

import numpy as np
import sonnet as snt


def make_default_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
) -> Mapping[str, types.TensorTransformation]:
  """Creates networks used by the agent."""

  # Get total number of action dimensions from action spec.
  num_dimensions = np.prod(action_spec.shape, dtype=int)

  policy_network = snt.Sequential([
      tf2_utils.batch_concat,
      networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
      networks.MultivariateNormalDiagHead(
          num_dimensions,
          tanh_mean=True,
          min_scale=0.3,
          init_scale=0.7,
          fixed_scale=False,
          use_tfd_independent=False)
  ])
  # The multiplexer concatenates the (maybe transformed) observations/actions.
  multiplexer = networks.CriticMultiplexer(
      action_network=networks.ClipToSpec(action_spec))
  critic_network = snt.Sequential([
      multiplexer,
      networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
      networks.NearZeroInitializedLinear(1),
  ])

  return {
      "policy": policy_network,
      "critic": critic_network,
  }


def make_network_with_prior(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (200, 100),
    critic_layer_sizes: Sequence[int] = (400, 300),
    prior_layer_sizes: Sequence[int] = (200, 100),
    policy_keys: Optional[Sequence[str]] = None,
    prior_keys: Optional[Sequence[str]] = None,
) -> Mapping[str, types.TensorTransformation]:
  """Creates networks used by the agent."""

  # Get total number of action dimensions from action spec.
  num_dimensions = np.prod(action_spec.shape, dtype=int)
  flatten_concat_policy = functools.partial(
      svg0_utils.batch_concat_selection, concat_keys=policy_keys)
  flatten_concat_prior = functools.partial(
      svg0_utils.batch_concat_selection, concat_keys=prior_keys)

  policy_network = snt.Sequential([
      flatten_concat_policy,
      networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
      networks.MultivariateNormalDiagHead(
          num_dimensions,
          tanh_mean=True,
          min_scale=0.1,
          init_scale=0.7,
          fixed_scale=False,
          use_tfd_independent=False)
  ])
  # The multiplexer concatenates the (maybe transformed) observations/actions.
  multiplexer = networks.CriticMultiplexer(
      observation_network=flatten_concat_policy,
      action_network=networks.ClipToSpec(action_spec))
  critic_network = snt.Sequential([
      multiplexer,
      networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
      networks.NearZeroInitializedLinear(1),
  ])
  prior_network = snt.Sequential([
      flatten_concat_prior,
      networks.LayerNormMLP(prior_layer_sizes, activate_final=True),
      networks.MultivariateNormalDiagHead(
          num_dimensions,
          tanh_mean=True,
          min_scale=0.1,
          init_scale=0.7,
          fixed_scale=False,
          use_tfd_independent=False)
  ])
  return {
      "policy": policy_network,
      "critic": critic_network,
      "prior": prior_network,
  }
