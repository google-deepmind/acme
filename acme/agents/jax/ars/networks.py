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

"""ARS networks definition."""

from typing import Tuple

from acme import specs
from acme.jax import networks as networks_lib
import jax.numpy as jnp


BEHAVIOR_PARAMS_NAME = 'policy'
EVAL_PARAMS_NAME = 'eval'


def make_networks(
    spec: specs.EnvironmentSpec) -> networks_lib.FeedForwardNetwork:
  """Creates networks used by the agent.

  The model used by the ARS paper is a simple clipped linear model.

  Args:
    spec: an environment spec

  Returns:
    A FeedForwardNetwork network.
  """

  obs_size = spec.observations.shape[0]
  act_size = spec.actions.shape[0]
  return networks_lib.FeedForwardNetwork(
      init=lambda _: jnp.zeros((obs_size, act_size)),
      apply=lambda matrix, obs: jnp.clip(jnp.matmul(obs, matrix), -1, 1))


def make_policy_network(
    network: networks_lib.FeedForwardNetwork,
    eval_mode: bool = True) -> Tuple[str, networks_lib.FeedForwardNetwork]:
  params_name = EVAL_PARAMS_NAME if eval_mode else BEHAVIOR_PARAMS_NAME
  return (params_name, network)
