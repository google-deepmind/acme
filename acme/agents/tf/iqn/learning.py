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

"""Implicit Quantile Network (IQN) learner implementation."""

from typing import Dict, List, Optional, Tuple

from acme import core
from acme import types
from acme.adders import reverb as adders
from acme.tf import losses
from acme.tf import networks
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf


class IQNLearner(core.Learner, tf2_savers.TFSaveable):
  """Distributional DQN learner."""

  def __init__(
      self,
      network: networks.IQNNetwork,
      target_network: snt.Module,
      discount: float,
      importance_sampling_exponent: float,
      learning_rate: float,
      target_update_period: int,
      dataset: tf.data.Dataset,
      huber_loss_parameter: float = 1.,
      replay_client: Optional[reverb.TFClient] = None,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      checkpoint: bool = True,
    ):
    """Initializes the learner.

    Args:
      network: the online Q network (the one being optimized) that outputs
        (q_values, q_logits, atoms).
      target_network: the target Q critic (which lags behind the online net).
      discount: discount to use for TD updates.
      importance_sampling_exponent: power to which importance weights are raised
        before normalizing.
      learning_rate: learning rate for the q-network update.
      target_update_period: number of learner steps to perform before updating
        the target networks.
      dataset: dataset to learn from, whether fixed or from a replay buffer
        (see `acme.datasets.reverb.make_reverb_dataset` documentation).
      huber_loss_parameter: Quadratic-linear boundary for Huber loss.
      replay_client: client to replay to allow for updating priorities.
      counter: Counter object for (potentially distributed) counting.
      logger: Logger object for writing logs to.
      checkpoint: boolean indicating whether to checkpoint the learner or not.
    """

    # Internalise agent components (replay buffer, networks, optimizer).
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

    # Create a snapshotter object.
    if checkpoint:
      self._checkpointer = tf2_savers.Checkpointer(
          time_delta_minutes=5,
          objects_to_save={
              'network': self._network,
              'target_network': self._target_network,
              'optimizer': self._optimizer,
              'num_steps': self._num_steps
          })
      self._snapshotter = tf2_savers.Snapshotter(
          objects_to_save={'network': network}, time_delta_minutes=60.)
    else:
      self._checkpointer = None
      self._snapshotter = None

  @tf.function
  def _step(self) -> Dict[str, tf.Tensor]:
    """Do a step of SGD and update the priorities."""

    # Pull out the data needed for updates/priorities.
    inputs = next(self._iterator)
    transitions: types.Transition = inputs.data
    keys, probs, *_ = inputs.info

    with tf.GradientTape() as tape:
      loss, fetches = self._loss_and_fetches(transitions.observation,
                                             transitions.action,
                                             transitions.reward,
                                             transitions.discount,
                                             transitions.next_observation)

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
      priorities = tf.clip_by_value(tf.abs(loss), -100, 100)
      priorities = tf.cast(priorities, tf.float64)
      self._replay_client.update_priorities(
          table=adders.DEFAULT_PRIORITY_TABLE, keys=keys, priorities=priorities)

    # Periodically update the target network.
    if tf.math.mod(self._num_steps, self._target_update_period) == 0:
      for src, dest in zip(self._network.variables,
                           self._target_network.variables):
        dest.assign(src)
    self._num_steps.assign_add(1)

    # Report gradient norms.
    fetches.update(
        loss=loss,
        gradient_norm=tf.linalg.global_norm(gradients))
    return fetches

  def step(self):
    # Do a batch of SGD.
    result = self._step()

    # Update our counts and record it.
    counts = self._counter.increment(steps=1)
    result.update(counts)

    # Checkpoint and attempt to write logs.
    if self._checkpointer is not None:
      self._checkpointer.save()
    if self._snapshotter is not None:
      self._snapshotter.save()
    self._logger.write(result)

  def get_variables(self, names: List[str]) -> List[np.ndarray]:
    return tf2_utils.to_numpy(self._variables)

  def _loss_and_fetches(
      self,
      o_tm1: tf.Tensor,
      a_tm1: tf.Tensor,
      r_t: tf.Tensor,
      d_t: tf.Tensor,
      o_t: tf.Tensor,
  ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    # Evaluate our networks.
    _, dist_tm1, tau = self._network(o_tm1)
    q_tm1 = _index_embs_with_actions(dist_tm1, a_tm1)

    q_selector, _, _ = self._target_network(o_t)
    a_t = tf.argmax(q_selector, axis=1)

    _, dist_t, _ = self._target_network(o_t)
    q_t = _index_embs_with_actions(dist_t, a_t)

    q_tm1 = losses.QuantileDistribution(values=q_tm1,
                                        logits=tf.zeros_like(q_tm1))
    q_t = losses.QuantileDistribution(values=q_t, logits=tf.zeros_like(q_t))

    # The rewards and discounts have to have the same type as network values.
    r_t = tf.cast(r_t, tf.float32)
    r_t = tf.clip_by_value(r_t, -1., 1.)
    d_t = tf.cast(d_t, tf.float32) * tf.cast(self._discount, tf.float32)

    # Compute the loss.
    loss_module = losses.NonUniformQuantileRegression(
        self._huber_loss_parameter)
    loss = loss_module(q_tm1, r_t, d_t, q_t, tau)

    # Compute statistics of the Q-values for logging.
    max_q = tf.reduce_max(q_t.values)
    min_q = tf.reduce_min(q_t.values)
    mean_q, var_q = tf.nn.moments(q_t.values, [0, 1])
    fetches = {
        'max_q': max_q,
        'mean_q': mean_q,
        'min_q': min_q,
        'var_q': var_q,
    }

    return loss, fetches

  @property
  def state(self):
    """Returns the stateful parts of the learner for checkpointing."""
    return {
        'network': self._network,
        'target_network': self._target_network,
        'optimizer': self._optimizer,
        'num_steps': self._num_steps
    }


def _index_embs_with_actions(
    embeddings: tf.Tensor,
    actions: tf.Tensor,
) -> tf.Tensor:
  """Slice an embedding Tensor with action indices.

  Take embeddings of the form [batch_size, num_actions, embed_dim]
  and actions of the form [batch_size], and return the sliced embeddings
  like embeddings[:, actions, :]. Doing this my way because the comments in
  the official op are scary.

  Args:
    embeddings: Tensor of embeddings to index.
    actions: int Tensor to use as index into embeddings

  Returns:
    Tensor of embeddings indexed by actions
  """
  batch_size, num_actions, _ = embeddings.shape.as_list()

  # Values are the 'values' in a sparse tensor we will be setting
  act_indx = tf.cast(actions, tf.int64)[:, None]
  values = tf.ones([tf.size(actions)], dtype=tf.bool)

  # Create a range for each index into the batch
  act_range = tf.range(0, batch_size, dtype=tf.int64)[:, None]
  # Combine this into coordinates with the action indices
  indices = tf.concat([act_range, act_indx], 1)

  actions_mask = tf.SparseTensor(indices, values, [batch_size, num_actions])
  actions_mask = tf.stop_gradient(
      tf.sparse.to_dense(actions_mask, default_value=False))
  sliced_emb = tf.boolean_mask(embeddings, actions_mask)
  return sliced_emb
