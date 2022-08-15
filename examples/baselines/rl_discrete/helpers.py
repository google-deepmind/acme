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
import os
from typing import Union

from absl import flags
from acme import core
from acme import environment_loop
from acme import specs
from acme import wrappers
from acme.agents.jax import builders
from acme.jax import experiments as experiments_lib
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.jax import utils
from acme.utils import counting
from acme.utils import experiment_utils
import atari_py  # pylint:disable=unused-import
import dm_env
import gym
import haiku as hk

FLAGS = flags.FLAGS


def make_atari_environment(
    level: str = 'Pong',
    sticky_actions: bool = True,
    zero_discount_on_life_loss: bool = False,
    oar_wrapper: bool = False,
) -> dm_env.Environment:
  """Loads the Atari environment."""
# Internal logic.
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
      wrappers.SinglePrecisionWrapper,
  ]

  if oar_wrapper:
    # E.g. IMPALA and R2D2 use this particular variant.
    wrapper_list.append(wrappers.ObservationActionRewardWrapper)

  return wrappers.wrap_all(env, wrapper_list)


def make_atari_evaluator_factory(
    level_name: str,
    network_factory: experiments_lib.NetworkFactory,
    agent_builder: Union[builders.ActorLearnerBuilder, builders.OfflineBuilder],
) -> experiments_lib.EvaluatorFactory:
  """Returns an Atari evaluator process."""

  def evaluator_factory(
      random_key: jax_types.PRNGKey,
      variable_source: core.VariableSource,
      counter: counting.Counter,
      make_actor: experiments_lib.MakeActorFn,
  ) -> environment_loop.EnvironmentLoop:
    """The evaluation process."""

    environment = make_atari_environment(
        level_name,
        sticky_actions=False,  # Turn off sticky actions for evaluation.
        oar_wrapper=True)
    environment_spec = specs.make_environment_spec(environment)
    networks = network_factory(environment_spec)
    policy = agent_builder.make_policy(
        networks, environment_spec, evaluation=True)
    actor = make_actor(random_key, policy, environment_spec, variable_source)

    # Create logger and counter.
    counter = counting.Counter(counter, 'evaluator')
    logger = experiment_utils.make_experiment_logger('evaluator', 'actor_steps')

    # Create the run loop and return it.
    return environment_loop.EnvironmentLoop(environment, actor, counter, logger)

  return evaluator_factory


def make_dqn_atari_network(
    environment_spec: specs.EnvironmentSpec) -> networks_lib.FeedForwardNetwork:
  """Creates networks for training DQN on Atari."""
  def network(inputs):
    model = hk.Sequential([
        networks_lib.AtariTorso(),
        hk.nets.MLP([512, environment_spec.actions.num_values]),
    ])
    return model(inputs)
  network_hk = hk.without_apply_rng(hk.transform(network))
  obs = utils.add_batch_dim(utils.zeros_like(environment_spec.observations))
  return networks_lib.FeedForwardNetwork(
      init=lambda rng: network_hk.init(rng, obs), apply=network_hk.apply)
