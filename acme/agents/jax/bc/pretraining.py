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

"""Tools to train a policy network with BC."""
from typing import Callable, Iterator

from acme import types
from acme.agents.jax.bc import learning
from acme.agents.jax.bc import losses
from acme.jax import networks as networks_lib
import jax
import optax


def train_with_bc(make_demonstrations: Callable[[int],
                                                Iterator[types.Transition]],
                  networks: networks_lib.FeedForwardNetwork,
                  loss: losses.Loss,
                  num_steps: int = 100000) -> networks_lib.Params:
  """Trains the given network with BC and returns the params.

  Args:
    make_demonstrations: A function (batch_size) -> iterator with demonstrations
      to be imitated.
    networks: Network taking (params, obs, is_training, key) as input
    loss: BC loss to use.
    num_steps: number of training steps

  Returns:
    The trained network params.
  """
  demonstration_iterator = make_demonstrations(256)

  learner = learning.BCLearner(
      network=networks, random_key=jax.random.PRNGKey(0), loss_fn=loss,
      demonstrations=demonstration_iterator, optimizer=optax.adam(1e-4),
      num_sgd_steps_per_step=1)

  # Train the agent
  for _ in range(num_steps):
    learner.step()

  return learner.get_variables(['policy'])[0]


def convert_to_bc_network(
    policy_network: networks_lib.FeedForwardNetwork
) -> networks_lib.FeedForwardNetwork:
  """Converts a policy_network from SAC/TD3/D4PG/.. into a BC policy network.

  Args:
    policy_network: FeedForwardNetwork taking the observation as input and
      returning action representation compatible with one of the BC losses.

  Returns:
    The BC policy network taking observation, is_training, key as input.
  """

  def apply(params, obs, is_training=False, key=None):
    del is_training, key
    return policy_network.apply(params, obs)

  return networks_lib.FeedForwardNetwork(policy_network.init, apply)


def convert_policy_value_to_bc_network(
    policy_value_network: networks_lib.FeedForwardNetwork
) -> networks_lib.FeedForwardNetwork:
  """Converts a network from e.g. PPO into a BC policy network.

  Args:
    policy_value_network: FeedForwardNetwork taking the observation as input.

  Returns:
    The BC policy network taking observation, is_training, key as input.
  """

  def apply(params, obs, is_training=False, key=None):
    del is_training, key
    actions, _ = policy_value_network.apply(params, obs)
    return actions

  return networks_lib.FeedForwardNetwork(policy_value_network.init, apply)
