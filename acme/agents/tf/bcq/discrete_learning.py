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

"""Discrete BCQ learner implementation.

As described in https://arxiv.org/pdf/1910.01708.pdf.
"""

import copy
from typing import Dict, List, Optional

from acme import core
from acme import types
from acme.adders import reverb as adders
from acme.agents.tf import bc
from acme.tf import losses
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.tf.networks import discrete as discrete_networks
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import trfl


class _InternalBCQLearner(core.Learner, tf2_savers.TFSaveable):
  """Internal BCQ learner.

  This implements the Q-learning component in the discrete BCQ algorithm.
  """

  def __init__(
      self,
      network: discrete_networks.DiscreteFilteredQNetwork,
      discount: float,
      importance_sampling_exponent: float,
      learning_rate: float,
      target_update_period: int,
      dataset: tf.data.Dataset,
      huber_loss_parameter: float = 1.,
      replay_client: Optional[reverb.TFClient] = None,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      checkpoint: bool = False,
  ):
    """Initializes the learner.

    Args:
      network: BCQ network
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
      checkpoint: boolean indicating whether to checkpoint the learner.
    """

    # Internalise agent components (replay buffer, networks, optimizer).
    # TODO(b/155086959): Fix type stubs and remove.
    self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types
    self._network = network
    self._q_network = network.q_network
    self._target_q_network = copy.deepcopy(network.q_network)
    self._optimizer = snt.optimizers.Adam(learning_rate)
    self._replay_client = replay_client

    # Internalise the hyperparameters.
    self._discount = discount
    self._target_update_period = target_update_period
    self._importance_sampling_exponent = importance_sampling_exponent
    self._huber_loss_parameter = huber_loss_parameter

    # Learner state.
    self._variables = [self._network.trainable_variables]
    self._num_steps = tf.Variable(0, dtype=tf.int32)

    # Internalise logging/counting objects.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger('learner',
                                                         save_data=False)

    # Create a snapshotter object.
    if checkpoint:
      self._snapshotter = tf2_savers.Snapshotter(
          objects_to_save={'network': network}, time_delta_minutes=60.)
    else:
      self._snapshotter = None

  @tf.function
  def _step(self) -> Dict[str, tf.Tensor]:
    """Do a step of SGD and update the priorities."""

    # Pull out the data needed for updates/priorities.
    inputs = next(self._iterator)
    transitions: types.Transition = inputs.data
    keys, probs = inputs.info[:2]

    with tf.GradientTape() as tape:
      # Evaluate our networks.
      q_tm1 = self._q_network(transitions.observation)
      q_t_value = self._target_q_network(transitions.next_observation)
      q_t_selector = self._network(transitions.next_observation)

      # The rewards and discounts have to have the same type as network values.
      r_t = tf.cast(transitions.reward, q_tm1.dtype)
      r_t = tf.clip_by_value(r_t, -1., 1.)
      d_t = tf.cast(transitions.discount, q_tm1.dtype) * tf.cast(
          self._discount, q_tm1.dtype)

      # Compute the loss.
      _, extra = trfl.double_qlearning(q_tm1, transitions.action, r_t, d_t,
                                       q_t_value, q_t_selector)
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
      for src, dest in zip(self._q_network.variables,
                           self._target_q_network.variables):
        dest.assign(src)
    self._num_steps.assign_add(1)

    # Compute the global norm of the gradients for logging.
    global_gradient_norm = tf.linalg.global_norm(gradients)

    # Compute statistics of the Q-values for logging.
    max_q = tf.reduce_max(q_t_value)
    min_q = tf.reduce_min(q_t_value)
    mean_q, var_q = tf.nn.moments(q_t_value, [0, 1])

    # Report loss & statistics for logging.
    fetches = {
        'gradient_norm': global_gradient_norm,
        'loss': loss,
        'max_q': max_q,
        'mean_q': mean_q,
        'min_q': min_q,
        'var_q': var_q,
    }

    return fetches

  def step(self):
    # Do a batch of SGD.
    result = self._step()

    # Update our counts and record it.
    counts = self._counter.increment(steps=1)
    result.update(counts)

    # Snapshot and attempt to write logs.
    if self._snapshotter is not None:
      self._snapshotter.save()
    self._logger.write(result)

  def get_variables(self, names: List[str]) -> List[np.ndarray]:
    return tf2_utils.to_numpy(self._variables)

  @property
  def state(self):
    """Returns the stateful parts of the learner for checkpointing."""
    return {
        'network': self._network,
        'target_q_network': self._target_q_network,
        'optimizer': self._optimizer,
        'num_steps': self._num_steps
    }


class DiscreteBCQLearner(core.Learner, tf2_savers.TFSaveable):
  """Discrete BCQ learner.

  This learner combines supervised BC learning and Q learning to implement the
  discrete BCQ algorithm as described in https://arxiv.org/pdf/1910.01708.pdf.
  """

  def __init__(self,
               network: discrete_networks.DiscreteFilteredQNetwork,
               dataset: tf.data.Dataset,
               learning_rate: float,
               counter: Optional[counting.Counter] = None,
               bc_logger: Optional[loggers.Logger] = None,
               bcq_logger: Optional[loggers.Logger] = None,
               **bcq_learner_kwargs):
    counter = counter or counting.Counter()
    self._bc_logger = bc_logger or loggers.TerminalLogger('bc_learner',
                                                          time_delta=1.)
    self._bcq_logger = bcq_logger or loggers.TerminalLogger('bcq_learner',
                                                            time_delta=1.)

    self._bc_learner = bc.BCLearner(
        network=network.g_network,
        learning_rate=learning_rate,
        dataset=dataset,
        counter=counting.Counter(counter, 'bc'),
        logger=self._bc_logger,
        checkpoint=False)
    self._bcq_learner = _InternalBCQLearner(
        network=network,
        learning_rate=learning_rate,
        dataset=dataset,
        counter=counting.Counter(counter, 'bcq'),
        logger=self._bcq_logger,
        **bcq_learner_kwargs)

  def get_variables(self, names):
    return self._bcq_learner.get_variables(names)

  @property
  def state(self):
    bc_state = self._bc_learner.state
    bc_state.pop('network')  # No need to checkpoint the BC network.
    bcq_state = self._bcq_learner.state
    state = dict()
    state.update({f'bc_{k}': v for k, v in bc_state.items()})
    state.update({f'bcq_{k}': v for k, v in bcq_state.items()})
    return state

  def step(self):
    self._bc_learner.step()
    self._bcq_learner.step()
