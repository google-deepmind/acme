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

"""Learner for the IMPALA actor-critic agent."""

import time
from typing import Dict, List, Mapping, Optional

import acme
from acme import specs
from acme.utils import counting
from acme.utils import loggers
from acme.utils import tf2_savers
from acme.utils import tf2_utils

import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree
import trfl

tfd = tfp.distributions


class IMPALALearner(acme.Learner, tf2_savers.TFSaveable):
  """Learner for an importanced-weighted advantage actor-critic."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      network: snt.RNNCore,
      dataset: tf.data.Dataset,
      learning_rate: float,
      discount: float = 0.99,
      entropy_cost: float = 0.,
      baseline_cost: float = 1.,
      max_abs_reward: Optional[float] = None,
      max_gradient_norm: Optional[float] = None,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
  ):

    # Internalise, optimizer, and dataset.
    self._env_spec = environment_spec
    self._optimizer = snt.optimizers.Adam(learning_rate=learning_rate)
    self._network = network
    self._variables = network.variables
    # TODO(b/155086959): Fix type stubs and remove.
    self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types

    # Hyperparameters.
    self._discount = discount
    self._entropy_cost = entropy_cost
    self._baseline_cost = baseline_cost

    # Set up reward/gradient clipping.
    if max_abs_reward is None:
      max_abs_reward = np.inf
    if max_gradient_norm is None:
      max_gradient_norm = 1e10  # A very large number. Infinity results in NaNs.
    self._max_abs_reward = tf.convert_to_tensor(max_abs_reward)
    self._max_gradient_norm = tf.convert_to_tensor(max_gradient_norm)

    # Set up logging/counting.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

    self._snapshotter = tf2_savers.Snapshotter(
        objects_to_save={'network': network}, time_delta_minutes=60.)

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

  @property
  def state(self) -> Mapping[str, tf2_savers.Checkpointable]:
    """Returns the stateful objects for checkpointing."""
    return {
        'network': self._network,
        'optimizer': self._optimizer,
    }

  @tf.function
  def _step(self) -> Dict[str, tf.Tensor]:
    """Does an SGD step on a batch of sequences."""

    # Retrieve a batch of data from replay.
    inputs: reverb.ReplaySample = next(self._iterator)
    data = tf2_utils.batch_to_sequence(inputs.data)
    observations, actions, rewards, discounts, extra = data
    core_state = tree.map_structure(lambda s: s[0], extra['core_state'])

    #
    actions = actions[:-1]  # [T-1]
    rewards = rewards[:-1]  # [T-1]
    discounts = discounts[:-1]  # [T-1]

    with tf.GradientTape() as tape:
      # Unroll current policy over observations.
      (logits, values), _ = snt.static_unroll(self._network, observations,
                                              core_state)

      # Compute importance sampling weights: current policy / behavior policy.
      behaviour_logits = extra['logits']
      pi_behaviour = tfd.Categorical(logits=behaviour_logits[:-1])
      pi_target = tfd.Categorical(logits=logits[:-1])
      log_rhos = pi_target.log_prob(actions) - pi_behaviour.log_prob(actions)

      # Optionally clip rewards.
      rewards = tf.clip_by_value(rewards,
                                 tf.cast(-self._max_abs_reward, rewards.dtype),
                                 tf.cast(self._max_abs_reward, rewards.dtype))

      # Critic loss.
      vtrace_returns = trfl.vtrace_from_importance_weights(
          log_rhos=tf.cast(log_rhos, tf.float32),
          discounts=tf.cast(self._discount * discounts, tf.float32),
          rewards=tf.cast(rewards, tf.float32),
          values=tf.cast(values[:-1], tf.float32),
          bootstrap_value=values[-1],
      )
      critic_loss = tf.square(vtrace_returns.vs - values[:-1])

      # Policy-gradient loss.
      policy_gradient_loss = trfl.policy_gradient(
          policies=pi_target,
          actions=actions,
          action_values=vtrace_returns.pg_advantages,
      )

      # Entropy regulariser.
      entropy_loss = trfl.policy_entropy_loss(pi_target).loss

      # Combine weighted sum of actor & critic losses.
      loss = tf.reduce_mean(policy_gradient_loss +
                            self._baseline_cost * critic_loss +
                            self._entropy_cost * entropy_loss)

    # Compute gradients and optionally apply clipping.
    gradients = tape.gradient(loss, self._network.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, self._max_gradient_norm)
    self._optimizer.apply(gradients, self._network.trainable_variables)

    metrics = {
        'loss': loss,
        'critic_loss': tf.reduce_mean(critic_loss),
        'entropy_loss': tf.reduce_mean(entropy_loss),
        'policy_gradient_loss': tf.reduce_mean(policy_gradient_loss),
    }

    return metrics

  def step(self):
    """Does a step of SGD and logs the results."""

    # Do a batch of SGD.
    results = self._step()

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Update our counts and record it.
    counts = self._counter.increment(steps=1, walltime=elapsed_time)
    results.update(counts)

    # Snapshot and attempt to write logs.
    self._snapshotter.save()
    self._logger.write(results)

  def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
    return [tf2_utils.to_numpy(self._variables)]
