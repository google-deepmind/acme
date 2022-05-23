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

"""Networks definitions for the BC agent."""

import dataclasses
import functools
from typing import Callable, Generic, Tuple, TypeVar

from acme import specs
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import jax.numpy as jnp


DirectRLNetworks = TypeVar('DirectRLNetworks')


@dataclasses.dataclass
class RNDNetworks(Generic[DirectRLNetworks]):
  """Container of RND networks factories."""
  target: networks_lib.FeedForwardNetwork
  predictor: networks_lib.FeedForwardNetwork
  # Function from predictor output, target output, and original reward to reward
  get_reward: Callable[
      [networks_lib.NetworkOutput, networks_lib.NetworkOutput, jnp.ndarray],
      jnp.ndarray]
  direct_rl_networks: DirectRLNetworks = None


# See Appendix A.2 of https://arxiv.org/pdf/1810.12894.pdf
def rnd_reward_fn(
    predictor_output: networks_lib.NetworkOutput,
    target_output: networks_lib.NetworkOutput,
    original_reward: jnp.ndarray,
    intrinsic_reward_coefficient: float = 1.0,
    extrinsic_reward_coefficient: float = 0.0,
) -> jnp.ndarray:
  intrinsic_reward = jnp.mean(
      jnp.square(predictor_output - target_output), axis=-1)
  return (intrinsic_reward_coefficient * intrinsic_reward +
          extrinsic_reward_coefficient * original_reward)


def make_networks(
    spec: specs.EnvironmentSpec,
    direct_rl_networks: DirectRLNetworks,
    layer_sizes: Tuple[int, ...] = (256, 256),
    intrinsic_reward_coefficient: float = 1.0,
    extrinsic_reward_coefficient: float = 0.0,
) -> RNDNetworks[DirectRLNetworks]:
  """Creates networks used by the agent and returns RNDNetworks.

  Args:
    spec: Environment spec.
    direct_rl_networks: Networks used by a direct rl algorithm.
    layer_sizes: Layer sizes.
    intrinsic_reward_coefficient: Multiplier on intrinsic reward.
    extrinsic_reward_coefficient: Multiplier on extrinsic reward.

  Returns:
    The RND networks.
  """

  def _rnd_fn(obs, act):
    # RND does not use the action but other variants like RED do.
    del act
    network = networks_lib.LayerNormMLP(list(layer_sizes))
    return network(obs)

  target = hk.without_apply_rng(hk.transform(_rnd_fn))
  predictor = hk.without_apply_rng(hk.transform(_rnd_fn))

  # Create dummy observations and actions to create network parameters.
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_obs = utils.add_batch_dim(dummy_obs)

  return RNDNetworks(
      target=networks_lib.FeedForwardNetwork(
          lambda key: target.init(key, dummy_obs, ()), target.apply),
      predictor=networks_lib.FeedForwardNetwork(
          lambda key: predictor.init(key, dummy_obs, ()), predictor.apply),
      direct_rl_networks=direct_rl_networks,
      get_reward=functools.partial(
          rnd_reward_fn,
          intrinsic_reward_coefficient=intrinsic_reward_coefficient,
          extrinsic_reward_coefficient=extrinsic_reward_coefficient))


def compute_rnd_reward(predictor_params: networks_lib.Params,
                       target_params: networks_lib.Params,
                       transitions: types.Transition,
                       networks: RNDNetworks) -> jnp.ndarray:
  """Computes the intrinsic RND reward for a given transition.

  Args:
    predictor_params: Parameters of the predictor network.
    target_params: Parameters of the target network.
    transitions: The sample to compute rewards for.
    networks: RND networks

  Returns:
    The rewards as an ndarray.
  """
  target_output = networks.target.apply(target_params, transitions.observation,
                                        transitions.action)
  predictor_output = networks.predictor.apply(predictor_params,
                                              transitions.observation,
                                              transitions.action)
  return networks.get_reward(predictor_output, target_output,
                             transitions.reward)
