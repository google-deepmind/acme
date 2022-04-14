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

"""ValueDice networks definition."""


import dataclasses
from typing import Callable, Optional, Tuple

from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass
class ValueDiceNetworks:
  """ValueDice networks."""
  policy_network: networks_lib.FeedForwardNetwork
  nu_network: networks_lib.FeedForwardNetwork
  # Functions for actors and evaluators, resp., to sample actions.
  sample: networks_lib.SampleFn
  sample_eval: Optional[networks_lib.SampleFn] = None
  # Function that transforms an action before a mixture is applied, typically
  # the identity for continuous actions and one-hot encoding for discrete
  # actions.
  encode_action: Callable[[networks_lib.Action], jnp.ndarray] = lambda x: x


def apply_policy_and_sample(
    networks: ValueDiceNetworks,
    eval_mode: bool = False) -> actor_core_lib.FeedForwardPolicy:
  """Returns a function that computes actions."""
  sample_fn = networks.sample if not eval_mode else networks.sample_eval
  if not sample_fn:
    raise ValueError('sample function is not provided')

  def apply_and_sample(params, key, obs):
    return sample_fn(networks.policy_network.apply(params, obs), key)
  return apply_and_sample


def make_networks(
    spec: specs.EnvironmentSpec,
    hidden_layer_sizes: Tuple[int, ...] = (256, 256)) -> ValueDiceNetworks:
  """Creates networks used by the agent."""

  num_dimensions = np.prod(spec.actions.shape, dtype=int)

  def _actor_fn(obs):
    network = hk.Sequential([
        hk.nets.MLP(
            list(hidden_layer_sizes),
            w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
            activation=jax.nn.relu,
            activate_final=True),
        networks_lib.NormalTanhDistribution(num_dimensions),
    ])
    return network(obs)

  def _nu_fn(obs, action):
    network = hk.Sequential([
        hk.nets.MLP(
            list(hidden_layer_sizes) + [1],
            w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
            activation=jax.nn.relu),
    ])
    return network(jnp.concatenate([obs, action], axis=-1))

  policy = hk.without_apply_rng(hk.transform(_actor_fn))
  nu = hk.without_apply_rng(hk.transform(_nu_fn))

  # Create dummy observations and actions to create network parameters.
  dummy_action = utils.zeros_like(spec.actions)
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_action = utils.add_batch_dim(dummy_action)
  dummy_obs = utils.add_batch_dim(dummy_obs)

  return ValueDiceNetworks(
      policy_network=networks_lib.FeedForwardNetwork(
          lambda key: policy.init(key, dummy_obs), policy.apply),
      nu_network=networks_lib.FeedForwardNetwork(
          lambda key: nu.init(key, dummy_obs, dummy_action), nu.apply),
      sample=lambda params, key: params.sample(seed=key),
      sample_eval=lambda params, key: params.mode())
