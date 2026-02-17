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

"""IQL learner implementation."""

import time
from typing import Dict, Iterator, NamedTuple, Optional

import acme
from acme import types
from acme.agents.jax.iql import networks as iql_networks
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import optax


class TrainingState(NamedTuple):
  """Contains training state for the IQL learner.
  
  Attributes:
    policy_optimizer_state: Optimizer state for policy network.
    value_optimizer_state: Optimizer state for value function network.
    critic_optimizer_state: Optimizer state for Q-function network.
    policy_params: Parameters of the policy network.
    value_params: Parameters of the value function network.
    critic_params: Parameters of the Q-function network.
    target_critic_params: Target network parameters for Q-function.
    key: Random number generator key.
    steps: Number of training steps completed.
  """
  policy_optimizer_state: optax.OptState
  value_optimizer_state: optax.OptState
  critic_optimizer_state: optax.OptState
  policy_params: networks_lib.Params
  value_params: networks_lib.Params
  critic_params: networks_lib.Params
  target_critic_params: networks_lib.Params
  key: networks_lib.PRNGKey
  steps: int = 0


class IQLLearner(acme.Learner):
  """IQL learner.

  Learning component of the Implicit Q-Learning algorithm from
  Kostrikov et al., 2021: https://arxiv.org/abs/2110.06169
  
  IQL is an offline RL algorithm that avoids querying values of out-of-sample
  actions by using expectile regression for the value function and advantage-
  weighted behavioral cloning for policy extraction.
  """

  _state: TrainingState

  def __init__(
      self,
      batch_size: int,
      networks: iql_networks.IQLNetworks,
      random_key: networks_lib.PRNGKey,
      demonstrations: Iterator[types.Transition],
      policy_optimizer: optax.GradientTransformation,
      value_optimizer: optax.GradientTransformation,
      critic_optimizer: optax.GradientTransformation,
      tau: float = 0.005,
      expectile: float = 0.7,
      temperature: float = 3.0,
      discount: float = 0.99,
      num_sgd_steps_per_step: int = 1,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None):
    """Initializes the IQL learner.

    Args:
      batch_size: Batch size for training.
      networks: IQL networks (policy, Q-function, value function).
      random_key: Random number generator key.
      demonstrations: Iterator over offline training data.
      policy_optimizer: Optimizer for policy network.
      value_optimizer: Optimizer for value function network.
      critic_optimizer: Optimizer for Q-function network.
      tau: Target network update coefficient (Polyak averaging).
      expectile: Expectile parameter for value function (0.5 = mean, >0.5 = upper expectile).
      temperature: Inverse temperature for advantage-weighted regression.
      discount: Discount factor for TD updates.
      num_sgd_steps_per_step: Number of gradient updates per step.
      counter: Counter for tracking training progress.
      logger: Logger for metrics.
    """
    self._batch_size = batch_size
    self._networks = networks
    self._demonstrations = demonstrations
    self._tau = tau
    self._expectile = expectile
    self._temperature = temperature
    self._discount = discount
    self._num_sgd_steps_per_step = num_sgd_steps_per_step
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

    # Initialize network parameters
    key_policy, key_value, key_critic, key = jax.random.split(random_key, 4)
    
    dummy_obs = utils.zeros_like(networks.environment_specs.observations)
    dummy_action = utils.zeros_like(networks.environment_specs.actions)
    
    policy_params = networks.policy_network.init(key_policy, dummy_obs)
    value_params = networks.value_network.init(key_value, dummy_obs)
    critic_params = networks.q_network.init(key_critic, dummy_obs, dummy_action)
    target_critic_params = critic_params

    # Initialize optimizers
    policy_optimizer_state = policy_optimizer.init(policy_params)
    value_optimizer_state = value_optimizer.init(value_params)
    critic_optimizer_state = critic_optimizer.init(critic_params)

    # Store state
    self._state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        value_optimizer_state=value_optimizer_state,
        critic_optimizer_state=critic_optimizer_state,
        policy_params=policy_params,
        value_params=value_params,
        critic_params=critic_params,
        target_critic_params=target_critic_params,
        key=key,
        steps=0)

    # Store optimizers
    self._policy_optimizer = policy_optimizer
    self._value_optimizer = value_optimizer
    self._critic_optimizer = critic_optimizer

    # Define update functions
    def expectile_loss(diff: jnp.ndarray, expectile: float) -> jnp.ndarray:
      """Asymmetric squared loss for expectile regression.
      
      Args:
        diff: Difference between target and prediction.
        expectile: Expectile parameter (0.5 = MSE, >0.5 = upper expectile).
      
      Returns:
        Expectile loss value.
      """
      weight = jnp.where(diff > 0, expectile, (1 - expectile))
      return weight * (diff ** 2)

    def value_loss_fn(
        value_params: networks_lib.Params,
        critic_params: networks_lib.Params,
        transitions: types.Transition) -> jnp.ndarray:
      """Computes value function loss using expectile regression.
      
      The value function is trained to approximate an upper expectile of the
      Q-values, which implicitly estimates the value of the best actions.
      
      Args:
        value_params: Value function parameters.
        critic_params: Q-function parameters (frozen during value update).
        transitions: Batch of transitions.
      
      Returns:
        Scalar loss value.
      """
      # Compute current Q-values
      q_values = networks.q_network.apply(
          critic_params, transitions.observation, transitions.action)
      q_values = jnp.min(q_values, axis=-1)  # Take minimum over ensemble
      
      # Compute value predictions
      v_pred = networks.value_network.apply(value_params, transitions.observation)
      v_pred = jnp.squeeze(v_pred, axis=-1)
      
      # Expectile regression loss
      diff = q_values - v_pred
      loss = expectile_loss(diff, self._expectile).mean()
      
      return loss

    def critic_loss_fn(
        critic_params: networks_lib.Params,
        value_params: networks_lib.Params,
        target_critic_params: networks_lib.Params,
        transitions: types.Transition) -> jnp.ndarray:
      """Computes Q-function loss using TD learning.
      
      The Q-function is trained with standard temporal difference learning,
      but uses the value function (instead of max Q) for the next state value.
      
      Args:
        critic_params: Q-function parameters.
        value_params: Value function parameters (frozen during Q update).
        target_critic_params: Target Q-function parameters.
        transitions: Batch of transitions.
      
      Returns:
        Scalar loss value.
      """
      # Compute next state values
      next_v = networks.value_network.apply(value_params, transitions.next_observation)
      next_v = jnp.squeeze(next_v, axis=-1)
      
      # Compute TD targets
      target_q = transitions.reward + self._discount * transitions.discount * next_v
      
      # Compute current Q predictions
      q_pred = networks.q_network.apply(
          critic_params, transitions.observation, transitions.action)
      
      # MSE loss
      loss = ((q_pred - jnp.expand_dims(target_q, -1)) ** 2).mean()
      
      return loss

    def policy_loss_fn(
        policy_params: networks_lib.Params,
        value_params: networks_lib.Params,
        critic_params: networks_lib.Params,
        transitions: types.Transition,
        key: networks_lib.PRNGKey) -> jnp.ndarray:
      """Computes policy loss using advantage-weighted regression.
      
      The policy is trained to maximize Q-values weighted by advantage,
      which is equivalent to behavioral cloning with advantage weights.
      
      Args:
        policy_params: Policy parameters.
        value_params: Value function parameters (frozen).
        critic_params: Q-function parameters (frozen).
        transitions: Batch of transitions.
        key: Random key (unused but kept for compatibility).
      
      Returns:
        Scalar loss value.
      """
      # Compute advantages
      v_values = networks.value_network.apply(value_params, transitions.observation)
      v_values = jnp.squeeze(v_values, axis=-1)
      
      q_values = networks.q_network.apply(
          critic_params, transitions.observation, transitions.action)
      q_values = jnp.min(q_values, axis=-1)
      
      advantages = q_values - v_values
      
      # Compute log probabilities
      dist_params = networks.policy_network.apply(policy_params, transitions.observation)
      log_probs = networks.log_prob(dist_params, transitions.action)
      
      # Advantage-weighted regression
      weights = jnp.exp(advantages * self._temperature)
      weights = jnp.minimum(weights, 100.0)  # Clip for numerical stability
      
      loss = -(weights * log_probs).mean()
      
      return loss

    # JIT compile update functions
    self._value_loss_fn = jax.jit(value_loss_fn)
    self._critic_loss_fn = jax.jit(critic_loss_fn)
    self._policy_loss_fn = jax.jit(policy_loss_fn)

    def value_update_step(
        state: TrainingState,
        transitions: types.Transition) -> TrainingState:
      """Performs one gradient step for value function."""
      value_loss, value_grads = jax.value_and_grad(value_loss_fn)(
          state.value_params, state.critic_params, transitions)
      
      value_updates, value_optimizer_state = self._value_optimizer.update(
          value_grads, state.value_optimizer_state)
      value_params = optax.apply_updates(state.value_params, value_updates)
      
      return state._replace(
          value_params=value_params,
          value_optimizer_state=value_optimizer_state)

    def critic_update_step(
        state: TrainingState,
        transitions: types.Transition) -> TrainingState:
      """Performs one gradient step for Q-function."""
      critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(
          state.critic_params, state.value_params,
          state.target_critic_params, transitions)
      
      critic_updates, critic_optimizer_state = self._critic_optimizer.update(
          critic_grads, state.critic_optimizer_state)
      critic_params = optax.apply_updates(state.critic_params, critic_updates)
      
      # Update target network
      target_critic_params = jax.tree_util.tree_map(
          lambda x, y: x * (1 - self._tau) + y * self._tau,
          state.target_critic_params, critic_params)
      
      return state._replace(
          critic_params=critic_params,
          critic_optimizer_state=critic_optimizer_state,
          target_critic_params=target_critic_params)

    def policy_update_step(
        state: TrainingState,
        transitions: types.Transition) -> TrainingState:
      """Performs one gradient step for policy."""
      policy_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(
          state.policy_params, state.value_params,
          state.critic_params, transitions, state.key)
      
      policy_updates, policy_optimizer_state = self._policy_optimizer.update(
          policy_grads, state.policy_optimizer_state)
      policy_params = optax.apply_updates(state.policy_params, policy_updates)
      
      return state._replace(
          policy_params=policy_params,
          policy_optimizer_state=policy_optimizer_state)

    self._value_update_step = jax.jit(value_update_step)
    self._critic_update_step = jax.jit(critic_update_step)
    self._policy_update_step = jax.jit(policy_update_step)

  def step(self):
    """Performs a single learner step (multiple gradient updates)."""
    for _ in range(self._num_sgd_steps_per_step):
      # Sample batch
      transitions = next(self._demonstrations)
      
      # Update value function
      self._state = self._value_update_step(self._state, transitions)
      
      # Update Q-function
      self._state = self._critic_update_step(self._state, transitions)
      
      # Update policy
      self._state = self._policy_update_step(self._state, transitions)
    
    # Increment step counter
    self._state = self._state._replace(steps=self._state.steps + 1)
    
    # Update counters and log
    counts = self._counter.increment(steps=1)
    
    # Periodically log metrics
    if self._state.steps % 100 == 0:
      # Compute losses for logging
      transitions = next(self._demonstrations)
      value_loss = self._value_loss_fn(
          self._state.value_params, self._state.critic_params, transitions)
      critic_loss = self._critic_loss_fn(
          self._state.critic_params, self._state.value_params,
          self._state.target_critic_params, transitions)
      policy_loss = self._policy_loss_fn(
          self._state.policy_params, self._state.value_params,
          self._state.critic_params, transitions, self._state.key)
      
      self._logger.write({
          'value_loss': float(value_loss),
          'critic_loss': float(critic_loss),
          'policy_loss': float(policy_loss),
          **counts
      })

  def get_variables(self, names: list[str]) -> list[networks_lib.Params]:
    """Returns network parameters."""
    variables = {
        'policy': self._state.policy_params,
        'critic': self._state.critic_params,
        'value': self._state.value_params,
    }
    return [variables[name] for name in names]

  def save(self) -> TrainingState:
    """Returns current training state for checkpointing."""
    return self._state

  def restore(self, state: TrainingState):
    """Restores training state from checkpoint."""
    self._state = state
