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

"""Network definitions for the IQL agent."""
import dataclasses
from typing import Optional

from acme import specs
from acme.agents.jax import sac
from acme.jax import networks as networks_lib


@dataclasses.dataclass
class IQLNetworks:
  """Networks and pure functions for the IQL agent.
  
  Attributes:
    policy_network: Policy network that outputs action distribution parameters.
    q_network: Q-function network that estimates state-action values.
    value_network: Value function network that estimates state values.
    log_prob: Function to compute log probability of actions.
    sample: Function to sample actions from policy.
    sample_eval: Function to sample actions for evaluation (typically deterministic).
    environment_specs: Environment specifications.
  """
  policy_network: networks_lib.FeedForwardNetwork
  q_network: networks_lib.FeedForwardNetwork
  value_network: networks_lib.FeedForwardNetwork
  log_prob: networks_lib.LogProbFn
  sample: Optional[networks_lib.SampleFn]
  sample_eval: Optional[networks_lib.SampleFn]
  environment_specs: specs.EnvironmentSpec


def make_networks(
    spec: specs.EnvironmentSpec,
    hidden_layer_sizes: tuple[int, ...] = (256, 256),
    **kwargs) -> IQLNetworks:
  """Creates networks for IQL agent.
  
  Args:
    spec: Environment specification.
    hidden_layer_sizes: Sizes of hidden layers for all networks.
    **kwargs: Additional arguments passed to SAC network creation.
  
  Returns:
    IQLNetworks containing policy, Q-function, and value function networks.
  """
  # Use SAC networks for policy and Q-function
  sac_networks = sac.make_networks(
      spec, 
      hidden_layer_sizes=hidden_layer_sizes,
      **kwargs)
  
  # Create value network (state -> scalar)
  action_spec = spec.actions
  observation_spec = spec.observations
  
  value_network = networks_lib.LayerNormMLP(
      layer_sizes=list(hidden_layer_sizes) + [1],
      activate_final=False)
  
  return IQLNetworks(
      policy_network=sac_networks.policy_network,
      q_network=sac_networks.q_network,
      value_network=value_network,
      log_prob=sac_networks.log_prob,
      sample=sac_networks.sample,
      sample_eval=sac_networks.sample_eval,
      environment_specs=spec)
