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

"""DQN agent implementation."""

import copy
from typing import Any, Callable, Iterable, Optional, Text

from acme import datasets
from acme import specs
from acme.adders import reverb as adders
from acme.open_spiel import open_spiel_specs
from acme.open_spiel.agents import agent
from acme.open_spiel.agents.tf import actors
from acme.open_spiel.agents.tf.dqn import learning
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import loggers
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import trfl


# TODO Import this from trfl once legal_actions_mask bug is fixed.
# See https://github.com/deepmind/trfl/pull/28
def epsilon_greedy(action_values, epsilon, legal_actions_mask=None):
  """Computes an epsilon-greedy distribution over actions.
  This returns a categorical distribution over a discrete action space. It is
  assumed that the trailing dimension of `action_values` is of length A, i.e.
  the number of actions. It is also assumed that actions are 0-indexed.
  This policy does the following:
  - With probability 1 - epsilon, take the action corresponding to the highest
  action value, breaking ties uniformly at random.
  - With probability epsilon, take an action uniformly at random.
  Args:
    action_values: A Tensor of action values with any rank >= 1 and dtype float.
      Shape can be flat ([A]), batched ([B, A]), a batch of sequences
      ([T, B, A]), and so on.
    epsilon: A scalar Tensor (or Python float) with value between 0 and 1.
    legal_actions_mask: An optional one-hot tensor having the shame shape and
      dtypes as `action_values`, defining the legal actions:
      legal_actions_mask[..., a] = 1 if a is legal, 0 otherwise.
      If not provided, all actions will be considered legal and
      `tf.ones_like(action_values)`.
  Returns:
    policy: tfp.distributions.Categorical distribution representing the policy.
  """
  # Convert inputs to Tensors if they aren't already.
  action_values = tf.convert_to_tensor(action_values)
  epsilon = tf.convert_to_tensor(epsilon, dtype=action_values.dtype)

  # We compute the action space dynamically.
  num_actions = tf.cast(tf.shape(action_values)[-1], action_values.dtype)

  if legal_actions_mask is None:
    # Dithering action distribution.
    dither_probs = 1 / num_actions * tf.ones_like(action_values)
    # Greedy action distribution, breaking ties uniformly at random.
    max_value = tf.reduce_max(action_values, axis=-1, keepdims=True)
    greedy_probs = tf.cast(tf.equal(action_values, max_value),
                           action_values.dtype)
  else:
    legal_actions_mask = tf.convert_to_tensor(legal_actions_mask)
    # Dithering action distribution.
    dither_probs = 1 / tf.reduce_sum(legal_actions_mask, axis=-1,
                                     keepdims=True) * legal_actions_mask
    masked_action_values = tf.where(tf.equal(legal_actions_mask, 1),
                                    action_values,
                                    tf.fill(tf.shape(action_values), -np.inf))
    # Greedy action distribution, breaking ties uniformly at random.
    max_value = tf.reduce_max(masked_action_values, axis=-1, keepdims=True)
    greedy_probs = tf.cast(
        tf.equal(action_values * legal_actions_mask, max_value),
        action_values.dtype)

  greedy_probs /= tf.reduce_sum(greedy_probs, axis=-1, keepdims=True)

  # Epsilon-greedy action distribution.
  probs = epsilon * dither_probs + (1 - epsilon) * greedy_probs

  # Make the policy object.
  policy = tfp.distributions.Categorical(probs=probs)

  return policy


# TODO Move to separate file.
class MaskedSequential(snt.Module):
  """Sonnet Module similar to Sequential but masks illegal actions."""

  def __init__(self,
               layers: Iterable[Callable[..., Any]] = None,
               epsilon: tf.Tensor = 0.0,
               name: Optional[Text] = None):
    super(MaskedSequential, self).__init__(name=name)
    self._layers = list(layers) if layers is not None else []
    self._epsilon = epsilon

  def __call__(self, inputs, legal_actions_mask):
    outputs = inputs
    for mod in self._layers:
      outputs = mod(outputs)
    outputs = epsilon_greedy(outputs, self._epsilon, legal_actions_mask)
    return outputs

class DQN(agent.OpenSpielAgent):
  """DQN agent.

  This implements a single-process DQN agent. This is a simple Q-learning
  algorithm that inserts N-step transitions into a replay buffer, and
  periodically updates its policy by sampling these transitions using
  prioritization.
  """

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      extras_spec: open_spiel_specs.ExtrasSpec,
      network: snt.Module,
      player_id: int,
      batch_size: int = 256,
      prefetch_size: int = 4,
      target_update_period: int = 100,
      samples_per_insert: float = 32.0,
      min_replay_size: int = 1000,
      max_replay_size: int = 1000000,
      importance_sampling_exponent: float = 0.2,
      priority_exponent: float = 0.6,
      n_step: int = 5,
      epsilon: Optional[tf.Tensor] = None,
      learning_rate: float = 1e-3,
      discount: float = 0.99,
      logger: loggers.Logger = None,
      checkpoint: bool = True,
      checkpoint_subpath: str = '~/acme/',
  ):
    """Initialize the agent.

    Args:
      environment_spec: description of the actions, observations, etc.
      network: the online Q network (the one being optimized)
      batch_size: batch size for updates.
      prefetch_size: size to prefetch from replay.
      target_update_period: number of learner steps to perform before updating
        the target networks.
      samples_per_insert: number of samples to take from replay for every insert
        that is made.
      min_replay_size: minimum replay size before updating. This and all
        following arguments are related to dataset construction and will be
        ignored if a dataset argument is passed.
      max_replay_size: maximum replay size.
      importance_sampling_exponent: power to which importance weights are raised
        before normalizing.
      priority_exponent: exponent used in prioritized sampling.
      n_step: number of steps to squash into a single transition.
      epsilon: probability of taking a random action; ignored if a policy
        network is given.
      learning_rate: learning rate for the q-network update.
      discount: discount to use for TD updates.
      logger: logger object to be used by learner.
      checkpoint: boolean indicating whether to checkpoint the learner.
      checkpoint_subpath: directory for the checkpoint.
    """

    # Create a replay server to add data to. This uses no limiter behavior in
    # order to allow the Agent interface to handle it.
    replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Prioritized(priority_exponent),
        remover=reverb.selectors.Fifo(),
        max_size=max_replay_size,
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=adders.NStepTransitionAdder.signature(environment_spec,
                                                        extras_spec))
    self._server = reverb.Server([replay_table], port=None)

    # The adder is used to insert observations into replay.
    address = f'localhost:{self._server.port}'
    adder = adders.NStepTransitionAdder(
        client=reverb.Client(address),
        n_step=n_step,
        discount=discount)

    # The dataset provides an interface to sample from replay.
    replay_client = reverb.TFClient(address)
    dataset = datasets.make_reverb_dataset(
        server_address=address,
        batch_size=batch_size,
        prefetch_size=prefetch_size)

    # Use constant 0.05 epsilon greedy policy by default.
    if epsilon is None:
      epsilon = tf.Variable(0.05, trainable=False)
    policy_network = MaskedSequential([network], epsilon)

    # Create a target network.
    target_network = copy.deepcopy(network)

    # Ensure that we create the variables before proceeding (maybe not needed).
    tf2_utils.create_variables(network, [environment_spec.observations])
    tf2_utils.create_variables(target_network, [environment_spec.observations])

    # Create the actor which defines how we take actions.
    actor = actors.FeedForwardActor(policy_network, adder)

    # The learner updates the parameters (and initializes them).
    learner = learning.DQNLearner(
        network=network,
        target_network=target_network,
        discount=discount,
        importance_sampling_exponent=importance_sampling_exponent,
        learning_rate=learning_rate,
        target_update_period=target_update_period,
        dataset=dataset,
        replay_client=replay_client,
        logger=logger,
        checkpoint=checkpoint)

    if checkpoint:
      self._checkpointer = tf2_savers.Checkpointer(
          directory=checkpoint_subpath,
          objects_to_save=learner.state,
          subdirectory='dqn_learner',
          time_delta_minutes=60.)
    else:
      self._checkpointer = None

    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=max(batch_size, min_replay_size),
        observations_per_step=float(batch_size) / samples_per_insert,
        player_id=player_id)

  def update(self):
    super().update()
    if self._checkpointer is not None:
      self._checkpointer.save()
