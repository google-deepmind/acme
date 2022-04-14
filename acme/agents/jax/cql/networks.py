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

"""Networks definitions for the CQL agent."""
import dataclasses
from typing import Optional, Tuple

from acme import specs
from acme.agents.jax import sac
from acme.jax import networks as networks_lib
import jax
import jax.numpy as jnp


@dataclasses.dataclass
class CQLNetworks:
  """Network and pure functions for the CQL agent."""
  policy_network: networks_lib.FeedForwardNetwork
  critic_network: networks_lib.FeedForwardNetwork
  log_prob: networks_lib.LogProbFn
  sample: Optional[networks_lib.SampleFn]
  sample_eval: Optional[networks_lib.SampleFn]
  environment_specs: specs.EnvironmentSpec


def apply_and_sample_n(key: networks_lib.PRNGKey,
                       networks: CQLNetworks,
                       params: networks_lib.Params, obs: jnp.ndarray,
                       num_samples: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Applies the policy and samples num_samples actions."""
  dist_params = networks.policy_network.apply(params, obs)
  sampled_actions = jnp.array([
      networks.sample(dist_params, key_n)
      for key_n in jax.random.split(key, num_samples)
  ])
  sampled_log_probs = networks.log_prob(dist_params, sampled_actions)
  return sampled_actions, sampled_log_probs


def make_networks(
    spec: specs.EnvironmentSpec, **kwargs) -> CQLNetworks:
  sac_networks = sac.make_networks(spec, **kwargs)
  return CQLNetworks(
      policy_network=sac_networks.policy_network,
      critic_network=sac_networks.q_network,
      log_prob=sac_networks.log_prob,
      sample=sac_networks.sample,
      sample_eval=sac_networks.sample_eval,
      environment_specs=spec)
