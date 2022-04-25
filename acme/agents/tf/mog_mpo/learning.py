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

"""Distributional MPO with MoG critic learner implementation."""

import dataclasses
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
import reverb
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


@dataclasses.dataclass
class PolicyEvaluationConfig:
  evaluate_stochastic_policy: bool = True
  num_value_samples: int = 128


class MoGMPOLearner(acme.Learner):
  """Distributional (MoG) MPO learner."""

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
      observation_network: snt.Module,
      target_observation_network: snt.Module,
      policy_evaluation_config: Optional[PolicyEvaluationConfig] = None,
      policy_loss_module: Optional[snt.Module] = None,
      policy_optimizer: Optional[snt.Optimizer] = None,
      critic_optimizer: Optional[snt.Optimizer] = None,
      dual_optimizer: Optional[snt.Optimizer] = None,
      clipping: bool = True,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      checkpoint: bool = True,
  ):

    # Store online and target networks.
    self._policy_network = policy_network
    self._critic_network = critic_network
    self._observation_network = observation_network
    self._target_policy_network = target_policy_network
    self._target_critic_network = target_critic_network
    self._target_observation_network = target_observation_network

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger('learner')

    # Other learner parameters.
    self._discount = discount
    self._num_samples = num_samples
    if policy_evaluation_config is None:
      policy_evaluation_config = PolicyEvaluationConfig()
    self._policy_evaluation_config = policy_evaluation_config
    self._clipping = clipping

    # Necessary to track when to update target networks.
    self._num_steps = tf.Variable(0, dtype=tf.int32)
    self._target_policy_update_period = target_policy_update_period
    self._target_critic_update_period = target_critic_update_period

    # Batch dataset and create iterator.
    self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

    self._policy_loss_module = policy_loss_module or losses.MPO(
        epsilon=1e-1,
        epsilon_mean=3e-3,
        epsilon_stddev=1e-6,
        epsilon_penalty=1e-3,
        init_log_temperature=10.,
        init_log_alpha_mean=10.,
        init_log_alpha_stddev=1000.)

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
          subdirectory='mog_mpo_learner',
          objects_to_save={
              'counter': self._counter,
              'policy': self._policy_network,
              'critic': self._critic_network,
              'observation': self._observation_network,
              'target_policy': self._target_policy_network,
              'target_critic': self._target_critic_network,
              'target_observation': self._target_observation_network,
              'policy_optimizer': self._policy_optimizer,
              'critic_optimizer': self._critic_optimizer,
              'dual_optimizer': self._dual_optimizer,
              'policy_loss_module': self._policy_loss_module,
              'num_steps': self._num_steps,
          })

      self._snapshotter = tf2_savers.Snapshotter(
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
  def _step(self, inputs: reverb.ReplaySample) -> types.NestedTensor:

    # Get data from replay (dropping extras if any). Note there is no
    # extra data here because we do not insert any into Reverb.
    o_tm1, a_tm1, r_t, d_t, o_t = (inputs.data.observation, inputs.data.action,
                                   inputs.data.reward, inputs.data.discount,
                                   inputs.data.next_observation)

    # Cast the additional discount to match the environment discount dtype.
    discount = tf.cast(self._discount, dtype=d_t.dtype)

    with tf.GradientTape(persistent=True) as tape:
      # Maybe transform the observation before feeding into policy and critic.
      # Transforming the observations this way at the start of the learning
      # step effectively means that the policy and critic share observation
      # network weights.
      o_tm1 = self._observation_network(o_tm1)
      # This stop_gradient prevents gradients to propagate into the target
      # observation network. In addition, since the online policy network is
      # evaluated at o_t, this also means the policy loss does not influence
      # the observation network training.
      o_t = tf.stop_gradient(self._target_observation_network(o_t))

      # Get online and target action distributions from policy networks.
      online_action_distribution = self._policy_network(o_t)
      target_action_distribution = self._target_policy_network(o_t)

      # Sample actions to evaluate policy; of size [N, B, ...].
      sampled_actions = target_action_distribution.sample(self._num_samples)

      # Tile embedded observations to feed into the target critic network.
      # Note: this is more efficient than tiling before the embedding layer.
      tiled_o_t = tf2_utils.tile_tensor(o_t, self._num_samples)  # [N, B, ...]

      # Compute target-estimated distributional value of sampled actions at o_t.
      sampled_q_t_distributions = self._target_critic_network(
          # Merge batch dimensions; to shape [N*B, ...].
          snt.merge_leading_dims(tiled_o_t, num_dims=2),
          snt.merge_leading_dims(sampled_actions, num_dims=2))

      # Compute online critic value distribution of a_tm1 in state o_tm1.
      q_tm1_distribution = self._critic_network(o_tm1, a_tm1)  # [B, ...]

      # Get the return distributions used in the policy evaluation bootstrap.
      if self._policy_evaluation_config.evaluate_stochastic_policy:
        z_distributions = sampled_q_t_distributions
        num_joint_samples = self._num_samples
      else:
        z_distributions = self._target_critic_network(
            o_t, target_action_distribution.mean())
        num_joint_samples = 1

      num_value_samples = self._policy_evaluation_config.num_value_samples
      num_joint_samples *= num_value_samples
      z_samples = z_distributions.sample(num_value_samples)
      z_samples = tf.reshape(z_samples, (num_joint_samples, -1, 1))

      # Expand dims of reward and discount tensors.
      reward = r_t[..., tf.newaxis]  # [B, 1]
      full_discount = discount * d_t[..., tf.newaxis]
      target_q = reward + full_discount * z_samples  # [N, B, 1]
      target_q = tf.stop_gradient(target_q)

      # Compute sample-based cross-entropy.
      log_probs_q = q_tm1_distribution.log_prob(target_q)  # [N, B, 1]
      critic_loss = -tf.reduce_mean(log_probs_q, axis=0)  # [B, 1]
      critic_loss = tf.reduce_mean(critic_loss)

      # Compute Q-values of sampled actions and reshape to [N, B].
      sampled_q_values = sampled_q_t_distributions.mean()
      sampled_q_values = tf.reshape(sampled_q_values, (self._num_samples, -1))

      # Compute MPO policy loss.
      policy_loss, policy_stats = self._policy_loss_module(
          online_action_distribution=online_action_distribution,
          target_action_distribution=target_action_distribution,
          actions=sampled_actions,
          q_values=sampled_q_values)
      policy_loss = tf.reduce_mean(policy_loss)

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
    # Log MPO stats.
    fetches.update(policy_stats)

    return fetches

  def step(self):
    self._maybe_update_target_networks()
    self._num_steps.assign_add(1)

    # Run the learning step.
    fetches = self._step(next(self._iterator))

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

  def _maybe_update_target_networks(self):
    # Update target network.
    online_policy_variables = self._policy_network.variables
    target_policy_variables = self._target_policy_network.variables
    online_critic_variables = (*self._observation_network.variables,
                               *self._critic_network.variables)
    target_critic_variables = (*self._target_observation_network.variables,
                               *self._target_critic_network.variables)

    # Make online policy -> target policy network update ops.
    if tf.math.mod(self._num_steps, self._target_policy_update_period) == 0:
      for src, dest in zip(online_policy_variables, target_policy_variables):
        dest.assign(src)

    # Make online critic -> target critic network update ops.
    if tf.math.mod(self._num_steps, self._target_critic_update_period) == 0:
      for src, dest in zip(online_critic_variables, target_critic_variables):
        dest.assign(src)
