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

"""Networks definition for CRR."""

import dataclasses
from typing import Callable, Tuple

from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass
class CRRNetworks:
  """Network and pure functions for the CRR agent.."""
  policy_network: networks_lib.FeedForwardNetwork
  critic_network: networks_lib.FeedForwardNetwork
  log_prob: networks_lib.LogProbFn
  sample: networks_lib.SampleFn
  sample_eval: networks_lib.SampleFn


def make_networks(
    spec: specs.EnvironmentSpec,
    policy_layer_sizes: Tuple[int, ...] = (256, 256),
    critic_layer_sizes: Tuple[int, ...] = (256, 256),
    activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
) -> CRRNetworks:
  """Creates networks used by the agent."""
  num_actions = np.prod(spec.actions.shape, dtype=int)

  # Create dummy observations and actions to create network parameters.
  dummy_action = utils.add_batch_dim(utils.zeros_like(spec.actions))
  dummy_obs = utils.add_batch_dim(utils.zeros_like(spec.observations))

  def _policy_fn(obs: jnp.ndarray) -> jnp.ndarray:
    network = hk.Sequential([
        hk.nets.MLP(
            list(policy_layer_sizes),
            w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
            activation=activation,
            activate_final=True),
        networks_lib.NormalTanhDistribution(num_actions),
    ])
    return network(obs)

  policy = hk.without_apply_rng(hk.transform(_policy_fn))
  policy_network = networks_lib.FeedForwardNetwork(
      lambda key: policy.init(key, dummy_obs), policy.apply)

  def _critic_fn(obs, action):
    network = hk.Sequential([
        hk.nets.MLP(
            list(critic_layer_sizes) + [1],
            w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
            activation=activation),
    ])
    data = jnp.concatenate([obs, action], axis=-1)
    return network(data)

  critic = hk.without_apply_rng(hk.transform(_critic_fn))
  critic_network = networks_lib.FeedForwardNetwork(
      lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply)

  return CRRNetworks(
      policy_network=policy_network,
      critic_network=critic_network,
      log_prob=lambda params, actions: params.log_prob(actions),
      sample=lambda params, key: params.sample(seed=key),
      sample_eval=lambda params, key: params.mode())
