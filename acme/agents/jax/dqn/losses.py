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

"""DQN losses."""
import dataclasses
from typing import Tuple

from acme import types
from acme.agents.jax.dqn import learning_lib
from acme.jax import networks as networks_lib
import jax
import jax.numpy as jnp
import reverb
import rlax


@dataclasses.dataclass
class PrioritizedDoubleQLearning(learning_lib.LossFn):
  """Clipped double q learning with prioritization on TD error."""
  discount: float = 0.99
  importance_sampling_exponent: float = 0.2
  max_abs_reward: float = 1.
  huber_loss_parameter: float = 1.

  def __call__(
      self,
      network: networks_lib.FeedForwardNetwork,
      params: networks_lib.Params,
      target_params: networks_lib.Params,
      batch: reverb.ReplaySample,
      key: networks_lib.PRNGKey,
  ) -> Tuple[jnp.DeviceArray, learning_lib.LossExtra]:
    """Calculate a loss on a single batch of data."""
    del key
    transitions: types.Transition = batch.data
    keys, probs, *_ = batch.info

    # Forward pass.
    q_tm1 = network.apply(params, transitions.observation)
    q_t_value = network.apply(target_params, transitions.next_observation)
    q_t_selector = network.apply(params, transitions.next_observation)

    # Cast and clip rewards.
    d_t = (transitions.discount * self.discount).astype(jnp.float32)
    r_t = jnp.clip(transitions.reward, -self.max_abs_reward,
                   self.max_abs_reward).astype(jnp.float32)

    # Compute double Q-learning n-step TD-error.
    batch_error = jax.vmap(rlax.double_q_learning)
    td_error = batch_error(q_tm1, transitions.action, r_t, d_t, q_t_value,
                           q_t_selector)
    batch_loss = rlax.huber_loss(td_error, self.huber_loss_parameter)

    # Importance weighting.
    importance_weights = (1. / probs).astype(jnp.float32)
    importance_weights **= self.importance_sampling_exponent
    importance_weights /= jnp.max(importance_weights)

    # Reweight.
    loss = jnp.mean(importance_weights * batch_loss)  # []
    reverb_update = learning_lib.ReverbUpdate(
        keys=keys, priorities=jnp.abs(td_error).astype(jnp.float64))
    extra = learning_lib.LossExtra(metrics={}, reverb_update=reverb_update)
    return loss, extra


@dataclasses.dataclass
class PrioritizedCategoricalDoubleQLearning(learning_lib.LossFn):
  """Categorical double q learning with prioritization on TD error."""
  discount: float = 0.99
  importance_sampling_exponent: float = 0.2
  max_abs_reward: float = 1.

  def __call__(
      self,
      network: networks_lib.FeedForwardNetwork,
      params: networks_lib.Params,
      target_params: networks_lib.Params,
      batch: reverb.ReplaySample,
      key: networks_lib.PRNGKey,
  ) -> Tuple[jnp.DeviceArray, learning_lib.LossExtra]:
    """Calculate a loss on a single batch of data."""
    del key
    transitions: types.Transition = batch.data
    keys, probs, *_ = batch.info

    # Forward pass.
    _, logits_tm1, atoms_tm1 = network.apply(params, transitions.observation)
    _, logits_t, atoms_t = network.apply(target_params,
                                         transitions.next_observation)
    q_t_selector, _, _ = network.apply(params, transitions.next_observation)

    # Cast and clip rewards.
    d_t = (transitions.discount * self.discount).astype(jnp.float32)
    r_t = jnp.clip(transitions.reward, -self.max_abs_reward,
                   self.max_abs_reward).astype(jnp.float32)

    # Compute categorical double Q-learning loss.
    batch_loss_fn = jax.vmap(
        rlax.categorical_double_q_learning,
        in_axes=(None, 0, 0, 0, 0, None, 0, 0))
    batch_loss = batch_loss_fn(atoms_tm1, logits_tm1, transitions.action, r_t,
                               d_t, atoms_t, logits_t, q_t_selector)

    # Importance weighting.
    importance_weights = (1. / probs).astype(jnp.float32)
    importance_weights **= self.importance_sampling_exponent
    importance_weights /= jnp.max(importance_weights)

    # Reweight.
    loss = jnp.mean(importance_weights * batch_loss)  # []
    reverb_update = learning_lib.ReverbUpdate(
        keys=keys, priorities=jnp.abs(batch_loss).astype(jnp.float64))
    extra = learning_lib.LossExtra(metrics={}, reverb_update=reverb_update)
    return loss, extra


@dataclasses.dataclass
class QLearning(learning_lib.LossFn):
  """Clipped q learning.

  This matches the original DQN loss: https://arxiv.org/abs/1312.5602.
  """
  discount: float = 0.99
  max_abs_reward: float = 1.
  huber_loss_parameter: float = 1.

  def __call__(
      self,
      network: networks_lib.FeedForwardNetwork,
      params: networks_lib.Params,
      target_params: networks_lib.Params,
      batch: reverb.ReplaySample,
      key: networks_lib.PRNGKey,
  ) -> Tuple[jnp.DeviceArray, learning_lib.LossExtra]:
    """Calculate a loss on a single batch of data."""
    del key
    transitions: types.Transition = batch.data

    # Forward pass.
    q_tm1 = network.apply(params, transitions.observation)
    q_t = network.apply(target_params, transitions.next_observation)

    # Cast and clip rewards.
    d_t = (transitions.discount * self.discount).astype(jnp.float32)
    r_t = jnp.clip(transitions.reward, -self.max_abs_reward,
                   self.max_abs_reward).astype(jnp.float32)

    # Compute Q-learning TD-error.
    batch_error = jax.vmap(rlax.q_learning)
    td_error = batch_error(q_tm1, transitions.action, r_t, d_t, q_t)
    batch_loss = rlax.huber_loss(td_error, self.huber_loss_parameter)

    loss = jnp.mean(batch_loss)
    extra = learning_lib.LossExtra(metrics={})
    return loss, extra


@dataclasses.dataclass
class RegularizedQLearning(learning_lib.LossFn):
  """Regularized Q-learning.

  Implements DQNReg loss function: https://arxiv.org/abs/2101.03958.
  This is almost identical to QLearning except: 1) Adds a regularization term;
  2) Uses vanilla TD error without huber loss. 3) No reward clipping.
  """
  discount: float = 0.99
  regularizer_coeff = 0.1

  def __call__(
      self,
      network: networks_lib.FeedForwardNetwork,
      params: networks_lib.Params,
      target_params: networks_lib.Params,
      batch: reverb.ReplaySample,
      key: networks_lib.PRNGKey,
  ) -> Tuple[jnp.DeviceArray, learning_lib.LossExtra]:
    """Calculate a loss on a single batch of data."""
    del key
    transitions: types.Transition = batch.data

    # Forward pass.
    q_tm1 = network.apply(params, transitions.observation)
    q_t = network.apply(target_params, transitions.next_observation)

    d_t = (transitions.discount * self.discount).astype(jnp.float32)

    # Compute Q-learning TD-error.
    batch_error = jax.vmap(rlax.q_learning)
    td_error = batch_error(
        q_tm1, transitions.action, transitions.reward, d_t, q_t)
    td_error = 0.5 * jnp.square(td_error)

    def select(qtm1, action):
      return qtm1[action]
    q_regularizer = jax.vmap(select)(q_tm1, transitions.action)

    loss = self.regularizer_coeff * jnp.mean(q_regularizer) + jnp.mean(td_error)
    extra = learning_lib.LossExtra(metrics={})
    return loss, extra


@dataclasses.dataclass
class MunchausenQLearning(learning_lib.LossFn):
  """Munchausen q learning.

  Implements M-DQN: https://arxiv.org/abs/2007.14430.
  """
  entropy_temperature: float = 0.03  # tau parameter
  munchausen_coefficient: float = 0.9  # alpha parameter
  clip_value_min: float = -1e3
  discount: float = 0.99
  max_abs_reward: float = 1.
  huber_loss_parameter: float = 1.

  def __call__(
      self,
      network: networks_lib.FeedForwardNetwork,
      params: networks_lib.Params,
      target_params: networks_lib.Params,
      batch: reverb.ReplaySample,
      key: networks_lib.PRNGKey,
  ) -> Tuple[jnp.DeviceArray, learning_lib.LossExtra]:
    """Calculate a loss on a single batch of data."""
    del key
    transitions: types.Transition = batch.data

    # Forward pass.
    q_online_s = network.apply(params, transitions.observation)
    action_one_hot = jax.nn.one_hot(transitions.action, q_online_s.shape[-1])
    q_online_sa = jnp.sum(action_one_hot * q_online_s, axis=-1)
    q_target_s = network.apply(target_params, transitions.observation)
    q_target_next = network.apply(target_params, transitions.next_observation)

    # Cast and clip rewards.
    d_t = (transitions.discount * self.discount).astype(jnp.float32)
    r_t = jnp.clip(transitions.reward, -self.max_abs_reward,
                   self.max_abs_reward).astype(jnp.float32)

    # Munchausen term : tau * log_pi(a|s)
    munchausen_term = self.entropy_temperature * jax.nn.log_softmax(
        q_target_s / self.entropy_temperature, axis=-1)
    munchausen_term_a = jnp.sum(action_one_hot * munchausen_term, axis=-1)
    munchausen_term_a = jnp.clip(munchausen_term_a,
                                 a_min=self.clip_value_min,
                                 a_max=0.)

    # Soft Bellman operator applied to q
    next_v = self.entropy_temperature * jax.nn.logsumexp(
        q_target_next / self.entropy_temperature, axis=-1)
    target_q = jax.lax.stop_gradient(r_t + self.munchausen_coefficient *
                                     munchausen_term_a + d_t * next_v)

    batch_loss = rlax.huber_loss(target_q - q_online_sa,
                                 self.huber_loss_parameter)
    loss = jnp.mean(batch_loss)

    extra = learning_lib.LossExtra(metrics={})
    return loss, extra
