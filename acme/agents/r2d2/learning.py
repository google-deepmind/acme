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

"""Recurrent Replay Distributed DQN (R2D2) learner implementation."""

import time
from typing import Dict, Iterator, List

import acme
from acme import losses
from acme import networks
from acme import specs
from acme.adders import reverb as adders
from acme.utils import counting
from acme.utils import loggers
from acme.utils import tf2_savers
from acme.utils import tf2_utils

import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import tree

Variables = List[np.ndarray]


class R2D2Learner(acme.Learner):
  """R2D2 learner.

  This is the learning component of the R2D2 agent. It takes a dataset as input
  and implements update functionality to learn from this dataset.
  """

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      network: networks.RNNCore,
      target_network: networks.RNNCore,
      burn_in_length: int,
      sequence_length: int,
      dataset: tf.data.Dataset,
      reverb_client: reverb.TFClient,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
      discount: float = 0.99,
      target_update_period: int = 100,
      importance_sampling_exponent: float = 0.2,
      max_replay_size: int = 1_000_000,
      learning_rate: float = 1e-3,
      store_lstm_state: bool = True,
      max_priority_weight: float = 0.9,
      n_step: int = 5,
      checkpoint: bool = True,
  ):

    # Internalise agent components (replay buffer, networks, optimizer).
    # TODO(b/155086959): Fix type stubs and remove.
    self._iterator: Iterator[reverb.ReplaySample] = iter(dataset)  # pytype: disable=wrong-arg-types
    self._network = network
    self._target_network = target_network
    self._optimizer = snt.optimizers.Adam(learning_rate, epsilon=1e-3)
    self._reverb_client = reverb_client

    # Internalise the hyperparameters.
    self._store_lstm_state = tf.constant(store_lstm_state)
    self._burn_in_length = burn_in_length
    self._discount = discount
    self._max_replay_size = max_replay_size
    self._importance_sampling_exponent = importance_sampling_exponent
    self._max_priority_weight = max_priority_weight
    self._target_update_period = target_update_period
    self._num_actions = environment_spec.actions.num_values
    self._sequence_length = sequence_length
    self._n_step = n_step

    # Learner state.
    self._variables = network.variables
    self._num_steps = tf.Variable(
        0., dtype=tf.float32, trainable=False, name='step')

    # Internalise logging/counting objects.
    self._counter = counting.Counter(counter, 'learner')
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=100.)

    # Checkpoint the networks.
    self._checkpointer = tf2_savers.Checkpointer(
        subdirectory='r2d2_learner',
        time_delta_minutes=5,
        enable_checkpointing=checkpoint,
        objects_to_save={
            'network': network,
            'target_network': target_network,
            'optimizer': self._optimizer,
            'num_steps': self._num_steps,
        })

    self._snapshotter = tf2_savers.Snapshotter(
        objects_to_save={'network': network}, time_delta_minutes=60.)

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

  @tf.function(experimental_compile=True)
  def _xla_step(self, sample: reverb.ReplaySample):
    data = tf2_utils.batch_to_sequence(sample.data)
    observations, actions, rewards, discounts, extra = data
    unused_sequence_length, batch_size = actions.shape

    # Get initial state for the LSTM, either from replay or simply use zeros.
    if self._store_lstm_state:
      core_state = tree.map_structure(lambda x: x[0], extra['core_state'])
    else:
      core_state = self._network.initial_state(batch_size)
    target_core_state = tree.map_structure(tf.identity, core_state)

    # Before training, optionally unroll the LSTM for a fixed warmup period.
    burn_in_obs = tree.map_structure(lambda x: x[:self._burn_in_length],
                                     observations)
    _, core_state = self._network.unroll(burn_in_obs, core_state,
                                         self._burn_in_length)
    _, target_core_state = self._target_network.unroll(burn_in_obs,
                                                       target_core_state,
                                                       self._burn_in_length)

    # Don't train on the warmup period.
    observations, actions, rewards, discounts, extra = tree.map_structure(
        lambda x: x[self._burn_in_length:],
        (observations, actions, rewards, discounts, extra))

    with tf.GradientTape() as tape:
      # Unroll the online and target Q-networks on the sequences.
      q_values, _ = self._network.unroll(observations, core_state,
                                         self._sequence_length)
      target_q_values, _ = self._target_network.unroll(observations,
                                                       target_core_state,
                                                       self._sequence_length)

      # Compute the target policy distribution (greedy).
      greedy_actions = tf.argmax(q_values, output_type=tf.int32, axis=-1)
      target_policy_probs = tf.one_hot(
          greedy_actions, depth=self._num_actions, dtype=q_values.dtype)

      # Compute the transformed n-step loss.
      rewards = tree.map_structure(lambda x: x[:-1], rewards)
      discounts = tree.map_structure(lambda x: x[:-1], discounts)
      loss, extra = losses.transformed_n_step_loss(
          qs=q_values,
          targnet_qs=target_q_values,
          actions=actions,
          rewards=rewards,
          pcontinues=discounts * self._discount,
          target_policy_probs=target_policy_probs,
          bootstrap_n=self._n_step,
      )

      # Calculate importance weights and use them to scale the loss.
      keys, probs, _ = sample.info
      probs = tf2_utils.batch_to_sequence(probs)
      importance_weights = 1. / (self._max_replay_size * probs)  # [T, B]
      importance_weights **= self._importance_sampling_exponent
      importance_weights /= tf.reduce_max(importance_weights)
      loss *= tf.cast(importance_weights, tf.float32)  # [T, B]
      loss = tf.reduce_mean(loss)  # []

    # Apply gradients via optimizer.
    gradients = tape.gradient(loss, self._network.trainable_variables)
    self._optimizer.apply(gradients, self._network.trainable_variables)

    # Periodically update the target network.
    if tf.math.mod(self._num_steps, self._target_update_period) == 0:
      for src, dest in zip(self._network.variables,
                           self._target_network.variables):
        dest.assign(src)
    self._num_steps.assign_add(1)

    # Compute updated priorities.
    priorities = compute_priority(extra.errors, self._max_priority_weight)

    return loss, keys[:, 0], priorities

  @tf.function
  def _step(self) -> Dict[str, tf.Tensor]:
    """Do a step of SGD and update the priorities."""

    # Pull out the data needed for updates/priorities.
    sample: reverb.ReplaySample = next(self._iterator)

    # Do a step of SGD.
    loss, keys, priorities = self._xla_step(sample)

    # Compute priorities and add an op to update them on the reverb side.
    self._reverb_client.update_priorities(
        table=adders.DEFAULT_PRIORITY_TABLE,
        keys=keys,
        priorities=tf.cast(priorities, tf.float64))

    return {'loss': loss}

  def step(self):
    # Run the learning step.
    results = self._step()

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Update our counts and record it.
    counts = self._counter.increment(steps=1, walltime=elapsed_time)
    results.update(counts)

    # Checkpoint and attempt to write the logs.
    self._checkpointer.save()
    self._snapshotter.save()
    self._logger.write(results)

  def get_variables(self, names: List[str]) -> List[Variables]:
    return [tf2_utils.to_numpy(self._variables)]


def compute_priority(errors: tf.Tensor, alpha: float):
  """Compute priority as mixture of max and mean sequence errors."""
  abs_errors = tf.abs(errors)
  mean_priority = tf.reduce_mean(abs_errors, axis=0)
  max_priority = tf.reduce_max(abs_errors, axis=0)

  return alpha * max_priority + (1 - alpha) * mean_priority
