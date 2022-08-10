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

"""Network definitions for BC."""

import dataclasses
from typing import Callable, Optional, Protocol

from acme.jax import networks as networks_lib
from acme.jax import types


class ApplyFn(Protocol):

  def __call__(self,
               params: networks_lib.Params,
               observation: networks_lib.Observation,
               *args,
               is_training: bool,
               key: Optional[types.PRNGKey] = None,
               **kwargs) -> networks_lib.NetworkOutput:
    ...


@dataclasses.dataclass
class BCPolicyNetwork:
  """Holds a pair of pure functions defining a policy network for BC.

  This is a feed-forward network taking params, obs, is_training, key as input.

  Attributes:
    init: A pure function. Initializes and returns the networks parameters.
    apply: A pure function. Computes and returns the outputs of a forward pass.
  """
  init: Callable[[types.PRNGKey], networks_lib.Params]
  apply: ApplyFn


def identity_sample(output: networks_lib.NetworkOutput,
                    key: types.PRNGKey) -> networks_lib.Action:
  """Placeholder sampling function for non-distributional networks."""
  del key
  return output


@dataclasses.dataclass
class BCNetworks:
  """The network and pure functions for the BC agent.

  Attributes:
    policy_network: The policy network.
    sample_fn: A pure function. Samples an action based on the network output.
      Must be set for distributional networks. Otherwise identity.
    log_prob: A pure function. Computes log-probability for an action.
      Must be set for distributional networks. Otherwise None.
  """
  policy_network: BCPolicyNetwork
  sample_fn: networks_lib.SampleFn = identity_sample
  log_prob: Optional[networks_lib.LogProbFn] = None


def convert_to_bc_network(
    policy_network: networks_lib.FeedForwardNetwork) -> BCPolicyNetwork:
  """Converts a policy network from SAC/TD3/D4PG/.. into a BC policy network.

  Args:
    policy_network: FeedForwardNetwork taking the observation as input and
      returning action representation compatible with one of the BC losses.

  Returns:
    The BC policy network taking observation, is_training, key as input.
  """

  def apply(params: networks_lib.Params,
            observation: networks_lib.Observation,
            *args,
            is_training: bool = False,
            key: Optional[types.PRNGKey] = None,
            **kwargs) -> networks_lib.NetworkOutput:
    del is_training, key
    return policy_network.apply(params, observation, *args, **kwargs)

  return BCPolicyNetwork(policy_network.init, apply)


def convert_policy_value_to_bc_network(
    policy_value_network: networks_lib.FeedForwardNetwork) -> BCPolicyNetwork:
  """Converts a policy-value network (e.g. from PPO) into a BC policy network.

  Args:
    policy_value_network: FeedForwardNetwork taking the observation as input.

  Returns:
    The BC policy network taking observation, is_training, key as input.
  """

  def apply(params: networks_lib.Params,
            observation: networks_lib.Observation,
            *args,
            is_training: bool = False,
            key: Optional[types.PRNGKey] = None,
            **kwargs) -> networks_lib.NetworkOutput:
    del is_training, key
    actions, _ = policy_value_network.apply(params, observation, *args,
                                            **kwargs)
    return actions

  return BCPolicyNetwork(policy_value_network.init, apply)
