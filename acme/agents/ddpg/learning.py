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

"""DDPG learner implementation."""

import time
from typing import List

import acme
from acme import losses
from acme import types
from acme.utils import counting
from acme.utils import loggers
from acme.utils import tf2_savers
from acme.utils import tf2_utils

import numpy as np
import sonnet as snt
import tensorflow as tf
import tree
import trfl


class DDPGLearner(acme.Learner):
  """DDPG learner.

  This is the learning component of a DDPG agent. IE it takes a dataset as input
  and implements update functionality to learn from this dataset.
  """

  def __init__(
      self,
      policy_network: snt.Module,
      critic_network: snt.Module,
      target_policy_network: snt.Module,
      target_critic_network: snt.Module,
      discount: float,
      target_update_period: int,
      dataset: tf.data.Dataset,
      observation_network: types.TensorTransformation = lambda x: x,
      target_observation_network: types.TensorTransformation = lambda x: x,
      policy_optimizer: snt.Optimizer = None,
      critic_optimizer: snt.Optimizer = None,
      clipping: bool = True,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
      checkpoint: bool = True,
  ):
    """Initializes the learner.

    Args:
      policy_network: the online (optimized) policy.
      critic_network: the online critic.
      target_policy_network: the target policy (which lags behind the online
        policy).
      target_critic_network: the target critic.
      discount: discount to use for TD updates.
      target_update_period: number of learner steps to perform before updating
        the target networks.
      dataset: dataset to learn from, whether fixed or from a replay buffer
        (see `acme.datasets.reverb.make_dataset` documentation).
      observation_network: an optional online network to process observations
        before the policy and the critic.
      target_observation_network: the target observation network.
      policy_optimizer: the optimizer to be applied to the DPG (policy) loss.
      critic_optimizer: the optimizer to be applied to the critic loss.
      clipping: whether to clip gradients by global norm.
      counter: counter object used to keep track of steps.
      logger: logger object to be used by learner.
      checkpoint: boolean indicating whether to checkpoint the learner.
    """

    # Store online and target networks.
    self._policy_network = policy_network
    self._critic_network = critic_network
    self._target_policy_network = target_policy_network
    self._target_critic_network = target_critic_network

    # Make sure observation networks are snt.Module's so they have variables.
    self._observation_network = tf2_utils.to_sonnet_module(observation_network)
    self._target_observation_network = tf2_utils.to_sonnet_module(
        target_observation_network)

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger('learner')

    # Other learner parameters.
    self._discount = discount
    self._clipping = clipping

    # Necessary to track when to update target networks.
    self._num_steps = tf.Variable(0, dtype=tf.int32)
    self._target_update_period = target_update_period

    # Create an iterator to go through the dataset.
    # TODO(b/155086959): Fix type stubs and remove.
    self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

    # Create optimizers if they aren't given.
    self._critic_optimizer = critic_optimizer or snt.optimizers.Adam(1e-4)
    self._policy_optimizer = policy_optimizer or snt.optimizers.Adam(1e-4)

    # Expose the variables.
    policy_network_to_expose = snt.Sequential(
        [self._target_observation_network, self._target_policy_network])
    self._variables = {
        'critic': target_critic_network.variables,
        'policy': policy_network_to_expose.variables,
    }

    self._checkpointer = tf2_savers.Checkpointer(
        time_delta_minutes=5,
        objects_to_save={
            'counter': self._counter,
            'policy': self._policy_network,
            'critic': self._critic_network,
            'target_policy': self._target_policy_network,
            'target_critic': self._target_critic_network,
            'policy_optimizer': self._policy_optimizer,
            'critic_optimizer': self._critic_optimizer,
            'num_steps': self._num_steps,
        },
        enable_checkpointing=checkpoint,
    )

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

  @tf.function
  def _step(self):
    # Update target network.
    online_variables = (
        *self._observation_network.variables,
        *self._critic_network.variables,
        *self._policy_network.variables,
    )
    target_variables = (
        *self._target_observation_network.variables,
        *self._target_critic_network.variables,
        *self._target_policy_network.variables,
    )

    # Make online -> target network update ops.
    if tf.math.mod(self._num_steps, self._target_update_period) == 0:
      for src, dest in zip(online_variables, target_variables):
        dest.assign(src)
    self._num_steps.assign_add(1)

    # Get data from replay (dropping extras if any). Note there is no
    # extra data here because we do not insert any into Reverb.
    inputs = next(self._iterator)
    o_tm1, a_tm1, r_t, d_t, o_t = inputs.data

    # Cast the additional discount to match the environment discount dtype.
    discount = tf.cast(self._discount, dtype=d_t.dtype)

    with tf.GradientTape(persistent=True) as tape:
      # Maybe transform the observation before feeding into policy and critic.
      # Transforming the observations this way at the start of the learning
      # step effectively means that the policy and critic share observation
      # network weights.
      o_tm1 = self._observation_network(o_tm1)
      o_t = self._target_observation_network(o_t)
      # This stop_gradient prevents gradients to propagate into the target
      # observation network. In addition, since the online policy network is
      # evaluated at o_t, this also means the policy loss does not influence
      # the observation network training.
      o_t = tree.map_structure(tf.stop_gradient, o_t)

      # Critic learning.
      q_tm1 = self._critic_network(o_tm1, a_tm1)
      q_t = self._target_critic_network(o_t, self._target_policy_network(o_t))

      # Squeeze into the shape expected by the td_learning implementation.
      q_tm1 = tf.squeeze(q_tm1, axis=-1)  # [B]
      q_t = tf.squeeze(q_t, axis=-1)  # [B]

      # Critic loss.
      critic_loss = trfl.td_learning(q_tm1, r_t, discount * d_t, q_t).loss
      critic_loss = tf.reduce_mean(critic_loss, axis=0)

      # Actor learning.
      dpg_a_t = self._policy_network(o_t)
      dpg_q_t = self._critic_network(o_t, dpg_a_t)

      # Actor loss. If clipping is true use dqda clipping and clip the norm.
      dqda_clipping = 1.0 if self._clipping else None
      policy_loss = losses.dpg(
          dpg_q_t,
          dpg_a_t,
          tape=tape,
          dqda_clipping=dqda_clipping,
          clip_norm=self._clipping)
      policy_loss = tf.reduce_mean(policy_loss, axis=0)

    # Get trainable variables.
    policy_variables = self._policy_network.trainable_variables
    critic_variables = (
        # In this agent, the critic loss trains the observation network.
        self._observation_network.trainable_variables +
        self._critic_network.trainable_variables)

    # Compute gradients.
    policy_gradients = tape.gradient(policy_loss, policy_variables)
    critic_gradients = tape.gradient(critic_loss, critic_variables)

    # Delete the tape manually because of the persistent=True flag.
    del tape

    # Maybe clip gradients.
    if self._clipping:
      policy_gradients = tf.clip_by_global_norm(policy_gradients, 40.)[0]
      critic_gradients = tf.clip_by_global_norm(critic_gradients, 40.)[0]

    # Apply gradients.
    self._policy_optimizer.apply(policy_gradients, policy_variables)
    self._critic_optimizer.apply(critic_gradients, critic_variables)

    # Losses to track.
    return {
        'critic_loss': critic_loss,
        'policy_loss': policy_loss,
    }

  def step(self):
    # Run the learning step.
    fetches = self._step()

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Update our counts and record it.
    counts = self._counter.increment(steps=1, walltime=elapsed_time)
    fetches.update(counts)

    # Checkpoint and attempt to write the logs.
    self._checkpointer.save()
    self._logger.write(fetches)

  def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
    return [tf2_utils.to_numpy(self._variables[name]) for name in names]
