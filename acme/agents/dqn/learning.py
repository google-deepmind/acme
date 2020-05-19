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

"""DQN learner implementation."""

import time
from typing import Dict, List

import acme
from acme import losses
from acme.adders import reverb as adders
from acme.utils import counting
from acme.utils import loggers
from acme.utils import tf2_savers
from acme.utils import tf2_utils

import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import trfl


class DQNLearner(acme.Learner, tf2_savers.TFSaveable):
  """DQN learner.

  This is the learning component of a DQN agent. It takes a dataset as input
  and implements update functionality to learn from this dataset. Optionally
  it takes a replay client as well to allow for updating of priorities.
  """

  def __init__(
      self,
      network: snt.Module,
      target_network: snt.Module,
      discount: float,
      importance_sampling_exponent: float,
      learning_rate: float,
      target_update_period: int,
      dataset: tf.data.Dataset,
      huber_loss_parameter: float = 1.,
      replay_client: reverb.TFClient = None,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
  ):
    """Initializes the learner.

    Args:
      network: the online Q network (the one being optimized)
      target_network: the target Q critic (which lags behind the online net).
      discount: discount to use for TD updates.
      importance_sampling_exponent: power to which importance weights are raised
        before normalizing.
      learning_rate: learning rate for the q-network update.
      target_update_period: number of learner steps to perform before updating
        the target networks.
      dataset: dataset to learn from, whether fixed or from a replay buffer (see
        `acme.datasets.reverb.make_dataset` documentation).
      huber_loss_parameter: Quadratic-linear boundary for Huber loss.
      replay_client: client to replay to allow for updating priorities.
      counter: Counter object for (potentially distributed) counting.
      logger: Logger object for writing logs to.
    """

    # Internalise agent components (replay buffer, networks, optimizer).
    # TODO(b/155086959): Fix type stubs and remove.
    self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types
    self._network = network
    self._target_network = target_network
    self._optimizer = snt.optimizers.Adam(learning_rate)
    self._replay_client = replay_client

    # Internalise the hyperparameters.
    self._discount = discount
    self._target_update_period = target_update_period
    self._importance_sampling_exponent = importance_sampling_exponent
    self._huber_loss_parameter = huber_loss_parameter

    # Learner state.
    self._variables: List[List[tf.Tensor]] = [network.trainable_variables]
    self._num_steps = tf.Variable(0, dtype=tf.int32)

    # Internalise logging/counting objects.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

    self._snapshotter = tf2_savers.Snapshotter(
        objects_to_save={'network': network}, time_delta_minutes=60.)

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

  @tf.function
  def _step(self) -> Dict[str, tf.Tensor]:
    """Do a step of SGD and update the priorities."""

    # Pull out the data needed for updates/priorities.
    inputs = next(self._iterator)
    o_tm1, a_tm1, r_t, d_t, o_t = inputs.data
    keys, probs = inputs.info[:2]

    with tf.GradientTape() as tape:
      # Evaluate our networks.
      q_tm1 = self._network(o_tm1)
      q_t_value = self._target_network(o_t)
      q_t_selector = self._network(o_t)

      # The rewards and discounts have to have the same type as network values.
      r_t = tf.cast(r_t, q_tm1.dtype)
      r_t = tf.clip_by_value(r_t, -1., 1.)
      d_t = tf.cast(d_t, q_tm1.dtype) * tf.cast(self._discount, q_tm1.dtype)

      # Compute the loss.
      _, extra = trfl.double_qlearning(q_tm1, a_tm1, r_t, d_t, q_t_value,
                                       q_t_selector)
      loss = losses.huber(extra.td_error, self._huber_loss_parameter)

      # Get the importance weights.
      importance_weights = 1. / probs  # [B]
      importance_weights **= self._importance_sampling_exponent
      importance_weights /= tf.reduce_max(importance_weights)

      # Reweight.
      loss *= tf.cast(importance_weights, loss.dtype)  # [B]
      loss = tf.reduce_mean(loss, axis=[0])  # []

    # Do a step of SGD.
    gradients = tape.gradient(loss, self._network.trainable_variables)
    self._optimizer.apply(gradients, self._network.trainable_variables)

    # Update the priorities in the replay buffer.
    if self._replay_client:
      priorities = tf.cast(tf.abs(extra.td_error), tf.float64)
      self._replay_client.update_priorities(
          table=adders.DEFAULT_PRIORITY_TABLE, keys=keys, priorities=priorities)

    # Periodically update the target network.
    if tf.math.mod(self._num_steps, self._target_update_period) == 0:
      for src, dest in zip(self._network.variables,
                           self._target_network.variables):
        dest.assign(src)
    self._num_steps.assign_add(1)

    # Report loss & statistics for logging.
    fetches = {
        'loss': loss,
    }

    return fetches

  def step(self):
    # Do a batch of SGD.
    result = self._step()

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Update our counts and record it.
    counts = self._counter.increment(steps=1, walltime=elapsed_time)
    result.update(counts)

    # Snapshot and attempt to write logs.
    self._snapshotter.save()
    self._logger.write(result)

  def get_variables(self, names: List[str]) -> List[np.ndarray]:
    return tf2_utils.to_numpy(self._variables)

  @property
  def state(self):
    """Returns the stateful parts of the learner for checkpointing."""
    return {
        'network': self._network,
        'target_network': self._target_network,
        'optimizer': self._optimizer,
        'num_steps': self._num_steps
    }
