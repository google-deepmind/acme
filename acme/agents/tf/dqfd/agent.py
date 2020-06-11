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

"""DQfD Agent implementation."""

import copy
import functools
import operator

from acme import datasets
from acme import specs
from acme import types as acme_types
from acme.adders import reverb as adders
from acme.agents import agent
from acme.agents.tf import actors
from acme.agents.tf import dqn
from acme.tf import utils as tf2_utils
import reverb
import sonnet as snt
import tensorflow as tf
import tree
import trfl


class DQfD(agent.Agent):
  """DQfD agent.

  This implements a single-process DQN agent that mixes demonstrations with
  actor experience.
  """

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      network: snt.Module,
      demonstration_dataset: tf.data.Dataset,
      demonstration_ratio: float,
      batch_size: int = 256,
      prefetch_size: int = 4,
      target_update_period: int = 100,
      samples_per_insert: float = 32.0,
      min_replay_size: int = 1000,
      max_replay_size: int = 1000000,
      importance_sampling_exponent: float = 0.2,
      n_step: int = 5,
      epsilon: tf.Tensor = None,
      learning_rate: float = 1e-3,
      discount: float = 0.99,
  ):
    """Initialize the agent.

    Args:
      environment_spec: description of the actions, observations, etc.
      network: the online Q network (the one being optimized)
      demonstration_dataset: tf.data.Dataset producing (timestep, action)
        tuples containing full episodes.
      demonstration_ratio: Ratio of transitions coming from demonstrations.
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
      n_step: number of steps to squash into a single transition.
      epsilon: probability of taking a random action; ignored if a policy
        network is given.
      learning_rate: learning rate for the q-network update.
      discount: discount to use for TD updates.
    """

    # Create a replay server to add data to. This uses no limiter behavior in
    # order to allow the Agent interface to handle it.
    replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=max_replay_size,
        rate_limiter=reverb.rate_limiters.MinSize(1))
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
        client=replay_client,
        environment_spec=environment_spec,
        transition_adder=True)

    # Combine with demonstration dataset.
    transition = functools.partial(_n_step_transition_from_episode,
                                   n_step=n_step,
                                   discount=discount)
    dataset_demos = demonstration_dataset.map(transition)
    dataset = tf.data.experimental.sample_from_datasets(
        [dataset, dataset_demos],
        [1 - demonstration_ratio, demonstration_ratio])

    # Batch and prefetch.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(prefetch_size)

    # Use constant 0.05 epsilon greedy policy by default.
    if epsilon is None:
      epsilon = tf.Variable(0.05, trainable=False)
    policy_network = snt.Sequential([
        network,
        lambda q: trfl.epsilon_greedy(q, epsilon=epsilon).sample(),
    ])

    # Create a target network.
    target_network = copy.deepcopy(network)

    # Ensure that we create the variables before proceeding (maybe not needed).
    tf2_utils.create_variables(network, [environment_spec.observations])
    tf2_utils.create_variables(target_network, [environment_spec.observations])

    # Create the actor which defines how we take actions.
    actor = actors.FeedForwardActor(policy_network, adder)

    # The learner updates the parameters (and initializes them).
    learner = dqn.DQNLearner(
        network=network,
        target_network=target_network,
        discount=discount,
        importance_sampling_exponent=importance_sampling_exponent,
        learning_rate=learning_rate,
        target_update_period=target_update_period,
        dataset=dataset,
        replay_client=replay_client)

    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=max(batch_size, min_replay_size),
        observations_per_step=float(batch_size) / samples_per_insert)


def _n_step_transition_from_episode(observations: acme_types.NestedTensor,
                                    actions: tf.Tensor,
                                    rewards: tf.Tensor,
                                    discounts: tf.Tensor,
                                    n_step: int,
                                    discount: float):
  """Produce Reverb-like N-step transition from a full episode.

  Observations, actions, rewards and discounts have the same length. This
  function will ignore the first reward and discount and the last action.

  Args:
    observations: [L, ...] Tensor.
    actions: [L, ...] Tensor.
    rewards: [L] Tensor.
    discounts: [L] Tensor.
    n_step: number of steps to squash into a single transition.
    discount: discount to use for TD updates.

  Returns:
    (o_t, a_t, r_t, d_t, o_tp1) tuple.
  """

  max_index = tf.shape(rewards)[0] - 1
  first = tf.random.uniform(shape=(), minval=0, maxval=max_index - 1,
                            dtype=tf.int32)
  last = tf.minimum(first + n_step, max_index)

  o_t = tree.map_structure(operator.itemgetter(first), observations)
  a_t = tree.map_structure(operator.itemgetter(first), actions)
  o_tp1 = tree.map_structure(operator.itemgetter(last), observations)

  # 0, 1, ..., n-1.
  discount_range = tf.cast(tf.range(last - first), tf.float32)
  # 1, g, ..., g^{n-1}.
  additional_discounts = tf.pow(discount, discount_range)
  # 1, d_t, d_t * d_{t+1}, ..., d_t * ... * d_{t+n-2}.
  discounts = tf.concat([[1.], tf.math.cumprod(discounts[first:last-1])], 0)
  # 1, g * d_t, ..., g^{n-1} * d_t * ... * d_{t+n-2}.
  discounts *= additional_discounts
  # r_t + g * d_t * r_{t+1} + ... + g^{n-1} * d_t * ... * d_{t+n-2} * r_{t+n-1}
  # We have to shift rewards by one so last=max_index corresponds to transitions
  # that include the last reward.
  r_t = tf.reduce_sum(rewards[first+1:last+1] * discounts)

  # g^{n-1} * d_{t} * ... * d_{t+n-1}.
  d_t = discounts[-1]

  key = tf.constant(0, tf.uint64)
  probability = tf.constant(1.0, tf.float64)
  table_size = tf.constant(1, tf.int64)
  info = reverb.SampleInfo(
      key=key, probability=probability, table_size=table_size)
  return reverb.ReplaySample(info=info, data=(o_t, a_t, r_t, d_t, o_tp1))
