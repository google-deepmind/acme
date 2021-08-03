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

"""Tests for DQN agent."""

from absl.testing import absltest
import acme
from acme import specs
from acme import wrappers
from acme.agents.jax import dqn
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.testing import fakes
import haiku as hk
import numpy as np

import bsuite
import dm_env
import gym
import functools

import jax

jax.config.update('jax_platform_name', "cpu")

raw_environment = bsuite.load_from_id(bsuite_id="catch/0")
environment = wrappers.SinglePrecisionWrapper(raw_environment)

# def make_environment(evaluation: bool = False,
#                      level: str = 'BreakoutNoFrameskip-v4') -> dm_env.Environment:
#   env = gym.make(level, full_action_space=True)
# 
#   max_episode_len = 108_000 if evaluation else 50_000
# 
#   return wrappers.wrap_all(env, [
#       wrappers.GymAtariAdapter,
#       functools.partial(
#           wrappers.AtariWrapper,
#           to_float=True,
#           max_episode_len=max_episode_len,
#           zero_discount_on_life_loss=True,
#       ),
#       wrappers.SinglePrecisionWrapper,
#   ])
# 
# environment = make_environment()

spec = specs.make_environment_spec(environment)

def network(x):
  model = hk.Sequential([
      hk.Flatten(),
      hk.nets.MLP([50, 50, spec.actions.num_values])
  ])
  return model(x)

# Make network purely functional
network_hk = hk.without_apply_rng(hk.transform(network, apply_rng=True))
dummy_obs = utils.add_batch_dim(utils.zeros_like(spec.observations))

# import pdb; pdb.set_trace()

network = networks_lib.FeedForwardNetwork(
    init=lambda rng: network_hk.init(rng, dummy_obs),
    apply=network_hk.apply)

class Printer():
  def __init__(self):
    self.returns = []
  def write(self, s):
    self.returns.append(s)
    print(s)

logger = Printer()

# Construct the agent.
agent = dqn.DQN(
    environment_spec=spec,
    network=network,
    samples_per_insert=32)

# Try running the environment loop. We have no assertions here because all
# we care about is that the agent runs without raising any errors.
loop = acme.EnvironmentLoop(environment, agent, logger=logger, should_update=True)


# jax.profiler.start_trace("/tmp/tensorboard")
# result = loop.run_episode() # is there some sort of ".block_until_ready()" we need here?
# jax.profiler.stop_trace()




# commented out for profiling
num_episodes=1000
loop.run(num_episodes=num_episodes)

# mean_sps = sum([x['steps_per_second'] for x in logger.returns])/len(logger.returns)

# print(f"{num_episodes} episodes completed. average steps per second of {mean_sps}.")

