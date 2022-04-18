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

"""SVG learner implementation."""

import time
from typing import Dict, Iterator, List, Optional

import acme
from acme.agents.tf.svg0_prior import utils as svg0_utils
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
from trfl import continuous_retrace_ops

_MIN_LOG_VAL = 1e-20


class SVG0Learner(acme.Learner):
  """SVG0 learner with optional prior.

  This is the learning component of an SVG0 agent. IE it takes a dataset as
  input and implements update functionality to learn from this dataset.
  """

  def __init__(
      self,
      policy_network: snt.Module,
      critic_network: snt.Module,
      target_policy_network: snt.Module,
      target_critic_network: snt.Module,
      discount: float,
      target_update_period: int,
      dataset_iterator: Iterator[reverb.ReplaySample],
      prior_network: Optional[snt.Module] = None,
      target_prior_network: Optional[snt.Module] = None,
      policy_optimizer: Optional[snt.Optimizer] = None,
      critic_optimizer: Optional[snt.Optimizer] = None,
      prior_optimizer: Optional[snt.Optimizer] = None,
      distillation_cost: Optional[float] = 1e-3,
      entropy_regularizer_cost: Optional[float] = 1e-3,
      num_action_samples: int = 10,
      lambda_: float = 1.0,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
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
      dataset_iterator: dataset to learn from, whether fixed or from a replay
        buffer (see `acme.datasets.reverb.make_reverb_dataset` documentation).
      prior_network: the online (optimized) prior.
      target_prior_network: the target prior (which lags behind the online
        prior).
      policy_optimizer: the optimizer to be applied to the SVG-0 (policy) loss.
      critic_optimizer: the optimizer to be applied to the distributional
        Bellman loss.
      prior_optimizer: the optimizer to be applied to the prior (distillation)
        loss.
      distillation_cost: a multiplier to be used when adding distillation
        against the prior to the losses.
      entropy_regularizer_cost: a multiplier used for per state sample based
        entropy added to the actor loss.
      num_action_samples: the number of action samples to use for estimating the
        value function and sample based entropy.
      lambda_: the `lambda` value to be used with retrace.
      counter: counter object used to keep track of steps.
      logger: logger object to be used by learner.
      checkpoint: boolean indicating whether to checkpoint the learner.
    """

    # Store online and target networks.
    self._policy_network = policy_network
    self._critic_network = critic_network
    self._target_policy_network = target_policy_network
    self._target_critic_network = target_critic_network

    self._prior_network = prior_network
    self._target_prior_network = target_prior_network

    self._lambda = lambda_
    self._num_action_samples = num_action_samples
    self._distillation_cost = distillation_cost
    self._entropy_regularizer_cost = entropy_regularizer_cost

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger('learner')

    # Other learner parameters.
    self._discount = discount

    # Necessary to track when to update target networks.
    self._num_steps = tf.Variable(0, dtype=tf.int32)
    self._target_update_period = target_update_period

    # Batch dataset and create iterator.
    self._iterator = dataset_iterator

    # Create optimizers if they aren't given.
    self._critic_optimizer = critic_optimizer or snt.optimizers.Adam(1e-4)
    self._policy_optimizer = policy_optimizer or snt.optimizers.Adam(1e-4)
    self._prior_optimizer = prior_optimizer or snt.optimizers.Adam(1e-4)

    # Expose the variables.
    self._variables = {
        'critic': self._critic_network.variables,
        'policy': self._policy_network.variables,
    }
    if self._prior_network is not None:
      self._variables['prior'] = self._prior_network.variables

    # Create a checkpointer and snapshotter objects.
    self._checkpointer = None
    self._snapshotter = None

    if checkpoint:
      objects_to_save = {
          'counter': self._counter,
          'policy': self._policy_network,
          'critic': self._critic_network,
          'target_policy': self._target_policy_network,
          'target_critic': self._target_critic_network,
          'policy_optimizer': self._policy_optimizer,
          'critic_optimizer': self._critic_optimizer,
          'num_steps': self._num_steps,
      }
      if self._prior_network is not None:
        objects_to_save['prior'] = self._prior_network
        objects_to_save['target_prior'] = self._target_prior_network
        objects_to_save['prior_optimizer'] = self._prior_optimizer

      self._checkpointer = tf2_savers.Checkpointer(
          subdirectory='svg0_learner',
          objects_to_save=objects_to_save)
      objects_to_snapshot = {
          'policy': self._policy_network,
          'critic': self._critic_network,
      }
      if self._prior_network is not None:
        objects_to_snapshot['prior'] = self._prior_network

      self._snapshotter = tf2_savers.Snapshotter(
          objects_to_save=objects_to_snapshot)

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

  @tf.function
  def _step(self) -> Dict[str, tf.Tensor]:
    # Update target network
    online_variables = [
        *self._critic_network.variables,
        *self._policy_network.variables,
    ]
    if self._prior_network is not None:
      online_variables += [*self._prior_network.variables]
    online_variables = tuple(online_variables)

    target_variables = [
        *self._target_critic_network.variables,
        *self._target_policy_network.variables,
    ]
    if self._prior_network is not None:
      target_variables += [*self._target_prior_network.variables]
    target_variables = tuple(target_variables)

    # Make online -> target network update ops.
    if tf.math.mod(self._num_steps, self._target_update_period) == 0:
      for src, dest in zip(online_variables, target_variables):
        dest.assign(src)
    self._num_steps.assign_add(1)

    # Get data from replay (dropping extras if any) and flip to `[T, B, ...]`.
    sample: reverb.ReplaySample = next(self._iterator)
    data = tf2_utils.batch_to_sequence(sample.data)
    observations, actions, rewards, discounts, extra = (data.observation,
                                                        data.action,
                                                        data.reward,
                                                        data.discount,
                                                        data.extras)
    online_target_pi_q = svg0_utils.OnlineTargetPiQ(
        online_pi=self._policy_network,
        online_q=self._critic_network,
        target_pi=self._target_policy_network,
        target_q=self._target_critic_network,
        num_samples=self._num_action_samples,
        online_prior=self._prior_network,
        target_prior=self._target_prior_network,
    )
    with tf.GradientTape(persistent=True) as tape:
      step_outputs = svg0_utils.static_rnn(
          core=online_target_pi_q,
          inputs=(observations, actions),
          unroll_length=rewards.shape[0])

      # Flip target samples to have shape [S, T+1, B, ...] where 'S' is the
      # number of action samples taken.
      target_pi_samples = tf2_utils.batch_to_sequence(
          step_outputs.target_samples)
      # Tile observations to have shape [S, T+1, B,..].
      tiled_observations = tf2_utils.tile_nested(observations,
                                                 self._num_action_samples)

      # Finally compute target Q values on the new action samples.
      # Shape: [S, T+1, B, 1]
      target_q_target_pi_samples = snt.BatchApply(self._target_critic_network,
                                                  3)(tiled_observations,
                                                     target_pi_samples)
      # Compute the value estimate by averaging over the action dimension.
      # Shape: [T+1, B, 1].
      target_v_target_pi = tf.reduce_mean(target_q_target_pi_samples, axis=0)

      # Split the target V's into the target for learning
      # `value_function_target` and the bootstrap value. Shape: [T, B].
      value_function_target = tf.squeeze(target_v_target_pi[:-1], axis=-1)
      # Shape: [B].
      bootstrap_value = tf.squeeze(target_v_target_pi[-1], axis=-1)

      # When learning with a prior, add entropy terms to value targets.
      if self._prior_network is not None:
        value_function_target -= self._distillation_cost * tf.stop_gradient(
            step_outputs.analytic_kl_to_target[:-1]
            )
        bootstrap_value -= self._distillation_cost * tf.stop_gradient(
            step_outputs.analytic_kl_to_target[-1])

      # Get target log probs and behavior log probs from rollout.
      # Shape: [T+1, B].
      target_log_probs_behavior_actions = (
          step_outputs.target_log_probs_behavior_actions)
      behavior_log_probs = extra['log_prob']
      # Calculate importance weights. Shape: [T+1, B].
      rhos = tf.exp(target_log_probs_behavior_actions - behavior_log_probs)

      # Filter the importance weights to mask out episode restarts. Ignore the
      # last action and consider the step type of the next step for masking.
      # Shape: [T, B].
      episode_start_mask = tf2_utils.batch_to_sequence(
          sample.data.start_of_episode)[1:]

      rhos = svg0_utils.mask_out_restarting(rhos[:-1], episode_start_mask)

      # rhos = rhos[:-1]
      # Compute the log importance weights with a small value added for
      # stability.
      # Shape: [T, B]
      log_rhos = tf.math.log(rhos + _MIN_LOG_VAL)

      # Retrieve the target and online Q values and throw away the last action.
      # Shape: [T, B].
      target_q_values = tf.squeeze(step_outputs.target_q[:-1], -1)
      online_q_values = tf.squeeze(step_outputs.online_q[:-1], -1)

      # Flip target samples to have shape [S, T+1, B, ...] where 'S' is the
      # number of action samples taken.
      online_pi_samples = tf2_utils.batch_to_sequence(
          step_outputs.online_samples)
      target_q_online_pi_samples = snt.BatchApply(self._target_critic_network,
                                                  3)(tiled_observations,
                                                     online_pi_samples)
      expected_q = tf.reduce_mean(
          tf.squeeze(target_q_online_pi_samples, -1), axis=0)

      # Flip online_log_probs to be of shape [S, T+1, B] and then compute
      # entropy by averaging over num samples. Final shape: [T+1, B].
      online_log_probs = tf2_utils.batch_to_sequence(
          step_outputs.online_log_probs)
      sample_based_entropy = tf.reduce_mean(-online_log_probs, axis=0)
      retrace_outputs = continuous_retrace_ops.retrace_from_importance_weights(
          log_rhos=log_rhos,
          discounts=self._discount * discounts[:-1],
          rewards=rewards[:-1],
          q_values=target_q_values,
          values=value_function_target,
          bootstrap_value=bootstrap_value,
          lambda_=self._lambda,
      )

      # Critic loss. Shape: [T, B].
      critic_loss = 0.5 * tf.math.squared_difference(
          tf.stop_gradient(retrace_outputs.qs), online_q_values)

      # Policy loss- SVG0 with sample based entropy. Shape: [T, B]
      policy_loss = -(
          expected_q + self._entropy_regularizer_cost * sample_based_entropy)
      policy_loss = policy_loss[:-1]

      if self._prior_network is not None:
        # When training the prior, also add the per-timestep KL cost.
        policy_loss += (
            self._distillation_cost * step_outputs.analytic_kl_to_target[:-1])

      # Ensure episode restarts are masked out when computing the losses.
      critic_loss = svg0_utils.mask_out_restarting(critic_loss,
                                                   episode_start_mask)
      critic_loss = tf.reduce_mean(critic_loss)

      policy_loss = svg0_utils.mask_out_restarting(policy_loss,
                                                   episode_start_mask)
      policy_loss = tf.reduce_mean(policy_loss)

      if self._prior_network is not None:
        prior_loss = step_outputs.analytic_kl_divergence[:-1]
        prior_loss = svg0_utils.mask_out_restarting(prior_loss,
                                                    episode_start_mask)
        prior_loss = tf.reduce_mean(prior_loss)

    # Get trainable variables.
    policy_variables = self._policy_network.trainable_variables
    critic_variables = self._critic_network.trainable_variables

    # Compute gradients.
    policy_gradients = tape.gradient(policy_loss, policy_variables)
    critic_gradients = tape.gradient(critic_loss, critic_variables)
    if self._prior_network is not None:
      prior_variables = self._prior_network.trainable_variables
      prior_gradients = tape.gradient(prior_loss, prior_variables)

    # Delete the tape manually because of the persistent=True flag.
    del tape

    # Apply gradients.
    self._policy_optimizer.apply(policy_gradients, policy_variables)
    self._critic_optimizer.apply(critic_gradients, critic_variables)
    losses = {
        'critic_loss': critic_loss,
        'policy_loss': policy_loss,
    }

    if self._prior_network is not None:
      self._prior_optimizer.apply(prior_gradients, prior_variables)
      losses['prior_loss'] = prior_loss

    # Losses to track.
    return losses

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
