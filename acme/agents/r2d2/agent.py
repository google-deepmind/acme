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

"""Recurrent DQN (R2D2) agent implementation."""

import copy

from acme import datasets
from acme import specs
from acme.adders import reverb as adders
from acme.agents import actors_tf2
from acme.agents import agent
from acme.agents.r2d2 import learning
from acme.utils import counting
from acme.utils import loggers
from acme.utils import tf2_savers
from acme.utils import tf2_utils

import reverb
import sonnet as snt
import tensorflow as tf
import trfl


class R2D2(agent.Agent):
  """R2D2 Agent.

  This implements a single-process R2D2 agent. This is a Q-learning algorithm
  that generates data via a (epislon-greedy) behavior policy, inserts
  trajectories into a replay buffer, and periodically updates the policy (and
  as a result the behavior) by sampling from this buffer.
  """

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      network: snt.RNNCore,
      burn_in_length: int,
      trace_length: int,
      replay_period: int,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
      discount: float = 0.99,
      batch_size: int = 32,
      prefetch_size: int = tf.data.experimental.AUTOTUNE,
      target_update_period: int = 100,
      importance_sampling_exponent: float = 0.2,
      priority_exponent: float = 0.6,
      epsilon: float = 0.01,
      learning_rate: float = 1e-3,
      min_replay_size: int = 1000,
      max_replay_size: int = 1000000,
      samples_per_insert: float = 32.0,
      store_lstm_state: bool = True,
      max_priority_weight: float = 0.9,
      checkpoint: bool = True,
  ):

    replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Prioritized(priority_exponent),
        remover=reverb.selectors.Fifo(),
        max_size=max_replay_size,
        rate_limiter=reverb.rate_limiters.MinSize(min_size_to_sample=1))
    self._server = reverb.Server([replay_table], port=None)
    address = f'localhost:{self._server.port}'

    sequence_length = burn_in_length + trace_length + 1
    # Component to add things into replay.
    adder = adders.SequenceAdder(
        client=reverb.Client(address),
        period=replay_period,
        sequence_length=sequence_length,
    )

    # The dataset object to learn from.
    reverb_client = reverb.TFClient(address)
    extra_spec = {
        'core_state': network.initial_state(1),
    }
    # Remove batch dimensions.
    extra_spec = tf2_utils.squeeze_batch_dim(extra_spec)
    dataset = datasets.make_reverb_dataset(
        client=reverb_client,
        environment_spec=environment_spec,
        batch_size=batch_size,
        prefetch_size=prefetch_size,
        extra_spec=extra_spec,
        sequence_length=sequence_length)

    target_network = copy.deepcopy(network)
    tf2_utils.create_variables(network, [environment_spec.observations])
    tf2_utils.create_variables(target_network, [environment_spec.observations])

    learner = learning.R2D2Learner(
        environment_spec=environment_spec,
        network=network,
        target_network=target_network,
        burn_in_length=burn_in_length,
        sequence_length=sequence_length,
        dataset=dataset,
        reverb_client=reverb_client,
        counter=counter,
        logger=logger,
        discount=discount,
        target_update_period=target_update_period,
        importance_sampling_exponent=importance_sampling_exponent,
        max_replay_size=max_replay_size,
        learning_rate=learning_rate,
        store_lstm_state=store_lstm_state,
        max_priority_weight=max_priority_weight,
    )

    self._checkpointer = tf2_savers.Checkpointer(
        subdirectory='r2d2_learner',
        time_delta_minutes=60,
        objects_to_save=learner.state,
        enable_checkpointing=checkpoint,
    )
    self._snapshotter = tf2_savers.Snapshotter(
        objects_to_save={'network': network}, time_delta_minutes=60.)

    policy_network = snt.DeepRNN([
        network,
        lambda qs: trfl.epsilon_greedy(qs, epsilon=epsilon).sample(),
    ])

    actor = actors_tf2.RecurrentActor(policy_network, adder)
    observations_per_step = (
        float(replay_period * batch_size) / samples_per_insert)
    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=replay_period * max(batch_size, min_replay_size),
        observations_per_step=observations_per_step)

  def update(self):
    super().update()
    self._snapshotter.save()
    self._checkpointer.save()
