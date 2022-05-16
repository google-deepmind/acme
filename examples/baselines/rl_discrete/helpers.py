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

"""Shared helpers for different discrete RL experiment flavours."""

import functools

from acme import specs
from acme import wrappers
from acme.jax import networks
from acme.jax import utils
import dm_env
import gym
import haiku as hk


def make_atari_environment(
    level: str = 'Pong',
    sticky_actions: bool = True,
    zero_discount_on_life_loss: bool = False) -> dm_env.Environment:
  """Loads the Atari environment."""
  version = 'v0' if sticky_actions else 'v4'
  level_name = f'{level}NoFrameskip-{version}'
  env = gym.make(level_name, full_action_space=True)

  wrapper_list = [
      wrappers.GymAtariAdapter,
      functools.partial(
          wrappers.AtariWrapper,
          to_float=True,
          max_episode_len=108_000,
          zero_discount_on_life_loss=zero_discount_on_life_loss,
      ),
  ]

  wrapper_list.append(wrappers.SinglePrecisionWrapper)

  return wrappers.wrap_all(env, wrapper_list)


def make_dqn_atari_network(
    environment_spec: specs.EnvironmentSpec) -> networks.FeedForwardNetwork:
  """Creates networks for training DQN on Atari."""
  def network(inputs):
    model = hk.Sequential([
        networks.AtariTorso(),
        hk.nets.MLP([512, environment_spec.actions.num_values]),
    ])
    return model(inputs)
  network_hk = hk.without_apply_rng(hk.transform(network))
  obs = utils.add_batch_dim(utils.zeros_like(environment_spec.observations))
  return networks.FeedForwardNetwork(
      init=lambda rng: network_hk.init(rng, obs),
      apply=network_hk.apply)
