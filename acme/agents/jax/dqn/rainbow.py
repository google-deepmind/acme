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

"""Defines Rainbow DQN, using JAX."""

import dataclasses
from typing import Callable


from acme import specs
from acme.agents.jax.dqn import actor as dqn_actor
from acme.agents.jax.dqn import builder
from acme.agents.jax.dqn import config as dqn_config
from acme.agents.jax.dqn import losses
from acme.jax import networks as networks_lib
from acme.jax import utils
import rlax

NetworkFactory = Callable[[specs.EnvironmentSpec],
                          networks_lib.FeedForwardNetwork]


@dataclasses.dataclass
class RainbowConfig(dqn_config.DQNConfig):
  """(Additional) configuration options for RainbowDQN."""
  max_abs_reward: float = 1.0  # For clipping reward


def apply_policy_and_sample(
    network: networks_lib.FeedForwardNetwork,) -> dqn_actor.EpsilonPolicy:
  """Returns a function that computes actions.

  Note that this differs from default_behavior_policy with that it
  expects c51-style network head which returns a tuple with the first entry
  representing q-values.

  Args:
    network: A c51-style feedforward network.

  Returns:
    A feedforward policy.
  """

  def apply_and_sample(params, key, obs, epsilon):
    # TODO(b/161332815): Make JAX Actor work with batched or unbatched inputs.
    obs = utils.add_batch_dim(obs)
    action_values = network.apply(params, obs)[0]
    action_values = utils.squeeze_batch_dim(action_values)
    return rlax.epsilon_greedy(epsilon).sample(key, action_values)

  return apply_and_sample


def eval_policy(network: networks_lib.FeedForwardNetwork,
                eval_epsilon: float) -> dqn_actor.EpsilonPolicy:
  """Returns a function that computes actions.

  Note that this differs from default_behavior_policy with that it
  expects c51-style network head which returns a tuple with the first entry
  representing q-values.

  Args:
    network: A c51-style feedforward network.
    eval_epsilon: for epsilon-greedy exploration.

  Returns:
    A feedforward policy.
  """
  policy = apply_policy_and_sample(network)

  def apply_and_sample(params, key, obs, _):
    return policy(params, key, obs, eval_epsilon)

  return apply_and_sample


def make_builder(config: RainbowConfig):
  """Returns a DQNBuilder with a pre-built loss function."""
  loss_fn = losses.PrioritizedCategoricalDoubleQLearning(
      discount=config.discount,
      importance_sampling_exponent=config.importance_sampling_exponent,
      max_abs_reward=config.max_abs_reward,
  )
  return builder.DQNBuilder(config, loss_fn=loss_fn)
