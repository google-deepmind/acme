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

from typing import Mapping, Sequence

from acme import specs
from acme.tf import networks
from acme.tf import utils as tf2_utils

import numpy as np
import sonnet as snt


def make_default_networks(
    environment_spec: specs.EnvironmentSpec,
    *,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    policy_init_scale: float = 0.7,
    critic_init_scale: float = 1e-3,
    critic_num_components: int = 5,
) -> Mapping[str, snt.Module]:
  """Creates networks used by the agent."""

  # Unpack the environment spec to get appropriate shapes, dtypes, etc.
  act_spec = environment_spec.actions
  obs_spec = environment_spec.observations
  num_dimensions = np.prod(act_spec.shape, dtype=int)

  # Create the observation network and make sure it's a Sonnet module.
  observation_network = tf2_utils.batch_concat
  observation_network = tf2_utils.to_sonnet_module(observation_network)

  # Create the policy network.
  policy_network = snt.Sequential([
      networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
      networks.MultivariateNormalDiagHead(
          num_dimensions,
          init_scale=policy_init_scale,
          use_tfd_independent=True)
  ])

  # The multiplexer concatenates the (maybe transformed) observations/actions.
  critic_network = snt.Sequential([
      networks.CriticMultiplexer(action_network=networks.ClipToSpec(act_spec)),
      networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
      networks.GaussianMixtureHead(
          num_dimensions=1,
          num_components=critic_num_components,
          init_scale=critic_init_scale)
  ])

  # Create network variables.
  # Get embedding spec by creating observation network variables.
  emb_spec = tf2_utils.create_variables(observation_network, [obs_spec])
  tf2_utils.create_variables(policy_network, [emb_spec])
  tf2_utils.create_variables(critic_network, [emb_spec, act_spec])

  return {
      'policy': policy_network,
      'critic': critic_network,
      'observation': observation_network,
  }
