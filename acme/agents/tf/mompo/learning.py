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

"""Multi-objective MPO learner implementation."""

import time
from typing import Callable, List, Optional, Sequence, Union

import acme
from acme import types
from acme.tf import losses
from acme.tf import networks
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import dataclasses
import numpy as np
import sonnet as snt
import tensorflow as tf
import trfl

ObjectiveFunctionSpec = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
RewardFunctionSpec = Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]

_DEFAULT_EPSILON = 1e-1
_DEFAULT_EPSILON_MEAN = 1e-3
_DEFAULT_EPSILON_STDDEV = 1e-6
_DEFAULT_INIT_LOG_TEMPERATURE = 1.
_DEFAULT_INIT_LOG_ALPHA_MEAN = 1.
_DEFAULT_INIT_LOG_ALPHA_STDDEV = 10.


@dataclasses.dataclass
class Objective:
  """Defines an objective for multi-objective MPO."""

  name: str
  # This computes "Q-values" directly from the sampled actions and other Q's.
  objective_fn: ObjectiveFunctionSpec


@dataclasses.dataclass
class Reward:
  """Defines an objective by specifying its reward function."""

  name: str
  # This computes the reward from observations, actions, and environment task
  # reward. In the learner, a head will automatically be added to the critic
  # network, to learn Q-values for this objective.
  reward_fn: RewardFunctionSpec


class MultiObjectiveMPOLearner(acme.Learner):
  """Distributional MPO learner.

  This is the learning component of a multi-objective MPO (MO-MPO) agent. A
  sequence of objectives must be specified. Otherwise, the inputs are identical
  to those of the MPO / DMPO learners.

  Each objective must be defined as either a Reward or an Objective. For each
  Reward, a critic will be trained to estimate Q-values for that objective.
  Whereas for each Objective, the Q-values are computed directly by that
  Objective's objective_fn.

  A Reward's reward_fn takes the observation, action, and environment reward as
  input, and returns the reward for that objective. For example, if the
  environment reward is a scalar, then an objective corresponding to the task
  would simply return the environment reward.

  An Objective's objective_fn takes the actions and reward-based objectives'
  Q-values as input, and outputs the "Q-values" for that objective. For
  instance, in the MO-MPO paper ([Abdolmaleki, Huang et al., 2020]), the action
  norm objective in the Humanoid run task is defined by setting the objective_fn
  to be the l2-norm of the actions.

  Note: If there is only one objective and that is the task reward, then this
  algorithm becomes exactly the same as (D)MPO.

  (Abdolmaleki, Huang et al., 2020): https://arxiv.org/pdf/2005.07513.pdf
  """

  def __init__(
      self,
      objectives: Sequence[Union[Objective, Reward]],
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
      policy_loss_module: Optional[losses.MultiObjectiveMPO] = None,
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
    self._num_samples = num_samples
    self._clipping = clipping

    # Necessary to track when to update target networks.
    self._num_steps = tf.Variable(0, dtype=tf.int32)
    self._target_policy_update_period = target_policy_update_period
    self._target_critic_update_period = target_critic_update_period

    # Batch dataset and create iterator.
    # TODO(b/155086959): Fix type stubs and remove.
    self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

    # Store objectives
    self._objectives = objectives
    self._num_objectives = len(objectives)  # K
    self._num_critic_heads = len(
        [x for x in objectives if hasattr(x, 'reward_fn')])  # C

    self._policy_loss_module = policy_loss_module or losses.MultiObjectiveMPO(
        epsilons=[losses.KLConstraint(x.name, _DEFAULT_EPSILON)
                  for x in self._objectives],
        epsilon_mean=_DEFAULT_EPSILON_MEAN,
        epsilon_stddev=_DEFAULT_EPSILON_STDDEV,
        init_log_temperature=_DEFAULT_INIT_LOG_TEMPERATURE,
        init_log_alpha_mean=_DEFAULT_INIT_LOG_ALPHA_MEAN,
        init_log_alpha_stddev=_DEFAULT_INIT_LOG_ALPHA_STDDEV)

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
          subdirectory='dmpo_learner',
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
  def _step(self) -> types.NestedTensor:
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

    self._num_steps.assign_add(1)

    # Get data from replay (dropping extras if any). Note there is no
    # extra data here because we do not insert any into Reverb.
    inputs = next(self._iterator)
    o_tm1, a_tm1, r_t, d_t, o_t = inputs.data

    # Get batch size and scalar dtype.
    batch_size = r_t.shape[0]

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
      sampled_q_t_all = self._target_critic_network(
          # Merge batch dimensions; to shape [N*B, ...].
          snt.merge_leading_dims(tiled_o_t, num_dims=2),
          snt.merge_leading_dims(sampled_actions, num_dims=2))

      # Compute online critic value distribution of a_tm1 in state o_tm1.
      q_tm1_all = self._critic_network(o_tm1, a_tm1)

      # Compute rewards for objectives with defined reward_fn
      reward_stats = {}
      r_t_all = []
      for objective in self._objectives:
        if hasattr(objective, 'reward_fn'):
          r = objective.reward_fn(o_tm1, a_tm1, r_t)
          reward_stats['{}_reward'.format(objective.name)] = tf.reduce_mean(r)
          r_t_all.append(r)
      r_t_all = tf.stack(r_t_all, axis=-1)
      r_t_all.get_shape().assert_has_rank(2)  # [B, C]

      if isinstance(sampled_q_t_all, list):  # Distributional critics
        # Compute average logits by first reshaping them and normalizing them
        # across atoms.
        critic_losses = []
        sampled_q_ts = []
        for idx, (sampled_q_t_distributions, q_tm1_distribution) in enumerate(
            zip(sampled_q_t_all, q_tm1_all)):
          # Compute loss for distributional critic for objective c
          sampled_logits = tf.reshape(
              sampled_q_t_distributions.logits,
              [self._num_samples, batch_size, -1])  # [N, B, A]
          sampled_logprobs = tf.math.log_softmax(sampled_logits, axis=-1)
          averaged_logits = tf.reduce_logsumexp(sampled_logprobs, axis=0)

          # Construct the expected distributional value for bootstrapping.
          q_t_distribution = networks.DiscreteValuedDistribution(
              values=sampled_q_t_distributions.values, logits=averaged_logits)

          # Compute critic distributional loss.
          critic_loss = losses.categorical(
              q_tm1_distribution, r_t_all[:, idx], discount * d_t,
              q_t_distribution)
          critic_losses.append(tf.reduce_mean(critic_loss))

          # Compute Q-values of sampled actions and reshape to [N, B].
          sampled_q_ts.append(tf.reshape(
              sampled_q_t_distributions.mean(), (self._num_samples, -1)))

        critic_loss = tf.reduce_mean(critic_losses)
        sampled_q_t = tf.stack(sampled_q_ts, axis=-1)  # [N, B, C]
      else:
        # Reshape Q-value samples back to original batch dimensions and average
        # them to compute the TD-learning bootstrap target.
        sampled_q_t = tf.reshape(
            sampled_q_t_all,
            (self._num_samples, batch_size, self._num_critic_heads))  # [N,B,C]
        q_t = tf.reduce_mean(sampled_q_t, axis=0)  # [B, C]

        # Flatten q_t and q_tm1; necessary for trfl.td_learning
        q_t = tf.reshape(q_t, [-1])  # [B*C]
        q_tm1 = tf.reshape(q_tm1_all, [-1])  # [B*C]

        # Flatten r_t_all; necessary for trfl.td_learning
        r_t_all = tf.reshape(r_t_all, [-1])  # [B*C]

        # Broadcast and then flatten d_t, to match shape of q_t and q_tm1
        d_t = tf.tile(d_t, [self._num_critic_heads])  # [B*C]

        # Critic loss.
        critic_loss = trfl.td_learning(q_tm1, r_t_all, discount * d_t, q_t).loss
        critic_loss = tf.reduce_mean(critic_loss)

      # Add sampled Q-values for objectives with defined objective_fn
      sampled_q_idx = 0
      sampled_q_t_k = []
      for objective in self._objectives:
        if hasattr(objective, 'reward_fn'):
          sampled_q_t_k.append(tf.stop_gradient(
              sampled_q_t[..., sampled_q_idx]))
          sampled_q_idx += 1
        if hasattr(objective, 'objective_fn'):
          sampled_q_t_k.append(tf.stop_gradient(
              objective.objective_fn(sampled_actions, sampled_q_t)))
      sampled_q_t_k = tf.stack(sampled_q_t_k, axis=-1)  # [N, B, K]

      # Compute MPO policy loss.
      policy_loss, policy_stats = self._policy_loss_module(
          online_action_distribution=online_action_distribution,
          target_action_distribution=target_action_distribution,
          actions=sampled_actions,
          q_values=sampled_q_t_k)

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
    fetches.update(reward_stats)  # Log reward stats.

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
