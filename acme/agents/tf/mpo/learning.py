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

"""MPO learner implementation."""

import time
from typing import List, Optional

import acme
from acme import types
from acme.tf import losses
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import sonnet as snt
import tensorflow as tf
import trfl


class MPOLearner(acme.Learner):
  """MPO learner."""

  def __init__(
      self,
      policy_network: snt.Module,
      critic_network: snt.Module,
      target_policy_network: snt.Module,
      target_critic_network: snt.Module,
      discount: float,
      num_samples: int,
      target_policy_update_period: int,
      target_critic_update_period: int,
      dataset: tf.data.Dataset,
      observation_network: types.TensorTransformation = tf.identity,
      target_observation_network: types.TensorTransformation = tf.identity,
      policy_loss_module: Optional[snt.Module] = None,
      policy_optimizer: Optional[snt.Optimizer] = None,
      critic_optimizer: Optional[snt.Optimizer] = None,
      dual_optimizer: Optional[snt.Optimizer] = None,
      clipping: bool = True,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      checkpoint: bool = True,
      save_directory: str = '~/acme/',
  ):

    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger('learner')
    self._discount = discount
    self._num_samples = num_samples
    self._clipping = clipping

    # Necessary to track when to update target networks.
    self._num_steps = tf.Variable(0, dtype=tf.int32)
    self._target_policy_update_period = target_policy_update_period
    self._target_critic_update_period = target_critic_update_period

    # Batch dataset and create iterator.
    # TODO(b/155086959): Fix type stubs and remove.
    self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

    # Store online and target networks.
    self._policy_network = policy_network
    self._critic_network = critic_network
    self._target_policy_network = target_policy_network
    self._target_critic_network = target_critic_network

    # Make sure observation networks are snt.Module's so they have variables.
    self._observation_network = tf2_utils.to_sonnet_module(observation_network)
    self._target_observation_network = tf2_utils.to_sonnet_module(
        target_observation_network)

    self._policy_loss_module = policy_loss_module or losses.MPO(
        epsilon=1e-1,
        epsilon_penalty=1e-3,
        epsilon_mean=1e-3,
        epsilon_stddev=1e-6,
        init_log_temperature=1.,
        init_log_alpha_mean=1.,
        init_log_alpha_stddev=10.)

    # Create the optimizers.
    self._critic_optimizer = critic_optimizer or snt.optimizers.Adam(1e-4)
    self._policy_optimizer = policy_optimizer or snt.optimizers.Adam(1e-4)
    self._dual_optimizer = dual_optimizer or snt.optimizers.Adam(1e-2)

    # Expose the variables.
    policy_network_to_expose = snt.Sequential(
        [self._target_observation_network, self._target_policy_network])
    self._variables = {
        'critic': self._target_critic_network.variables,
        'policy': policy_network_to_expose.variables,
    }

    # Create a checkpointer and snapshotter object.
    self._checkpointer = None
    self._snapshotter = None

    if checkpoint:
      self._checkpointer = tf2_savers.Checkpointer(
          directory=save_directory,
          subdirectory='mpo_learner',
          objects_to_save={
              'counter': self._counter,
              'policy': self._policy_network,
              'critic': self._critic_network,
              'observation_network': self._observation_network,
              'target_policy': self._target_policy_network,
              'target_critic': self._target_critic_network,
              'target_observation_network': self._target_observation_network,
              'policy_optimizer': self._policy_optimizer,
              'critic_optimizer': self._critic_optimizer,
              'dual_optimizer': self._dual_optimizer,
              'policy_loss_module': self._policy_loss_module,
              'num_steps': self._num_steps,
          })

      self._snapshotter = tf2_savers.Snapshotter(
          directory=save_directory,
          objects_to_save={
              'policy':
                  snt.Sequential([
                      self._target_observation_network,
                      self._target_policy_network
                  ]),
          })

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

  @tf.function
  def _step(self) -> types.Nest:
    # Update target network.
    online_policy_variables = self._policy_network.variables
    target_policy_variables = self._target_policy_network.variables
    online_critic_variables = (
        *self._observation_network.variables,
        *self._critic_network.variables,
    )
    target_critic_variables = (
        *self._target_observation_network.variables,
        *self._target_critic_network.variables,
    )

    # Make online policy -> target policy network update ops.
    if tf.math.mod(self._num_steps, self._target_policy_update_period) == 0:
      for src, dest in zip(online_policy_variables, target_policy_variables):
        dest.assign(src)
    # Make online critic -> target critic network update ops.
    if tf.math.mod(self._num_steps, self._target_critic_update_period) == 0:
      for src, dest in zip(online_critic_variables, target_critic_variables):
        dest.assign(src)

    # Increment number of learner steps for periodic update bookkeeping.
    self._num_steps.assign_add(1)

    # Get next batch of data.
    inputs = next(self._iterator)

    # Get data from replay (dropping extras if any). Note there is no
    # extra data here because we do not insert any into Reverb.
    transitions: types.Transition = inputs.data

    # Cast the additional discount to match the environment discount dtype.
    discount = tf.cast(self._discount, dtype=transitions.discount.dtype)

    with tf.GradientTape(persistent=True) as tape:
      # Maybe transform the observation before feeding into policy and critic.
      # Transforming the observations this way at the start of the learning
      # step effectively means that the policy and critic share observation
      # network weights.
      o_tm1 = self._observation_network(transitions.observation)
      # This stop_gradient prevents gradients to propagate into the target
      # observation network. In addition, since the online policy network is
      # evaluated at o_t, this also means the policy loss does not influence
      # the observation network training.
      o_t = tf.stop_gradient(
          self._target_observation_network(transitions.next_observation))

      # Get action distributions from policy networks.
      online_action_distribution = self._policy_network(o_t)
      target_action_distribution = self._target_policy_network(o_t)

      # Get sampled actions to evaluate policy; of size [N, B, ...].
      sampled_actions = target_action_distribution.sample(self._num_samples)
      tiled_o_t = tf2_utils.tile_tensor(o_t, self._num_samples)  # [N, B, ...]

      # Compute the target critic's Q-value of the sampled actions in state o_t.
      sampled_q_t = self._target_critic_network(
          # Merge batch dimensions; to shape [N*B, ...].
          snt.merge_leading_dims(tiled_o_t, num_dims=2),
          snt.merge_leading_dims(sampled_actions, num_dims=2))

      # Reshape Q-value samples back to original batch dimensions and average
      # them to compute the TD-learning bootstrap target.
      sampled_q_t = tf.reshape(sampled_q_t, (self._num_samples, -1))  # [N, B]
      q_t = tf.reduce_mean(sampled_q_t, axis=0)  # [B]

      # Compute online critic value of a_tm1 in state o_tm1.
      q_tm1 = self._critic_network(o_tm1, transitions.action)  # [B, 1]
      q_tm1 = tf.squeeze(q_tm1, axis=-1)  # [B]; necessary for trfl.td_learning.

      # Critic loss.
      critic_loss = trfl.td_learning(q_tm1, transitions.reward,
                                     discount * transitions.discount, q_t).loss
      critic_loss = tf.reduce_mean(critic_loss)

      # Actor learning.
      policy_loss, policy_stats = self._policy_loss_module(
          online_action_distribution=online_action_distribution,
          target_action_distribution=target_action_distribution,
          actions=sampled_actions,
          q_values=sampled_q_t)

    # For clarity, explicitly define which variables are trained by which loss.
    critic_trainable_variables = (
        # In this agent, the critic loss trains the observation network.
        self._observation_network.trainable_variables +
        self._critic_network.trainable_variables)
    policy_trainable_variables = self._policy_network.trainable_variables
    # The following are the MPO dual variables, stored in the loss module.
    dual_trainable_variables = self._policy_loss_module.trainable_variables

    # Compute gradients.
    critic_gradients = tape.gradient(critic_loss, critic_trainable_variables)
    policy_gradients, dual_gradients = tape.gradient(
        policy_loss, (policy_trainable_variables, dual_trainable_variables))

    # Delete the tape manually because of the persistent=True flag.
    del tape

    # Maybe clip gradients.
    if self._clipping:
      policy_gradients = tuple(tf.clip_by_global_norm(policy_gradients, 40.)[0])
      critic_gradients = tuple(tf.clip_by_global_norm(critic_gradients, 40.)[0])

    # Apply gradients.
    self._critic_optimizer.apply(critic_gradients, critic_trainable_variables)
    self._policy_optimizer.apply(policy_gradients, policy_trainable_variables)
    self._dual_optimizer.apply(dual_gradients, dual_trainable_variables)

    # Losses to track.
    fetches = {
        'critic_loss': critic_loss,
        'policy_loss': policy_loss,
    }
    fetches.update(policy_stats)  # Log MPO stats.

    return fetches

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
    if self._checkpointer is not None:
      self._checkpointer.save()
    if self._snapshotter is not None:
      self._snapshotter.save()
    self._logger.write(fetches)

  def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
    return [tf2_utils.to_numpy(self._variables[name]) for name in names]
