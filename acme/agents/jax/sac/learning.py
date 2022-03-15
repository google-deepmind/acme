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

"""SAC learner implementation."""

import time
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple

import acme
from acme import types
from acme.agents.jax.sac import networks as sac_networks
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import optax
import reverb


class TrainingState(NamedTuple):
  """Contains training state for the learner."""
  policy_optimizer_state: optax.OptState
  q_optimizer_state: optax.OptState
  policy_params: networks_lib.Params
  q_params: networks_lib.Params
  target_q_params: networks_lib.Params
  key: networks_lib.PRNGKey
  alpha_optimizer_state: Optional[optax.OptState] = None
  alpha_params: Optional[networks_lib.Params] = None


class SACLearner(acme.Learner):
  """SAC learner."""

  _state: TrainingState

  def __init__(
      self,
      networks: sac_networks.SACNetworks,
      rng: jnp.ndarray,
      iterator: Iterator[reverb.ReplaySample],
      policy_optimizer: optax.GradientTransformation,
      q_optimizer: optax.GradientTransformation,
      tau: float = 0.005,
      reward_scale: float = 1.0,
      discount: float = 0.99,
      entropy_coefficient: Optional[float] = None,
      target_entropy: float = 0,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      num_sgd_steps_per_step: int = 1):
    """Initialize the SAC learner.

    Args:
      networks: SAC networks
      rng: a key for random number generation.
      iterator: an iterator over training data.
      policy_optimizer: the policy optimizer.
      q_optimizer: the Q-function optimizer.
      tau: target smoothing coefficient.
      reward_scale: reward scale.
      discount: discount to use for TD updates.
      entropy_coefficient: coefficient applied to the entropy bonus. If None, an
        adaptative coefficient will be used.
      target_entropy: Used to normalize entropy. Only used when
        entropy_coefficient is None.
      counter: counter object used to keep track of steps.
      logger: logger object to be used by learner.
      num_sgd_steps_per_step: number of sgd steps to perform per learner 'step'.
    """
    adaptive_entropy_coefficient = entropy_coefficient is None
    if adaptive_entropy_coefficient:
      # alpha is the temperature parameter that determines the relative
      # importance of the entropy term versus the reward.
      log_alpha = jnp.asarray(0., dtype=jnp.float32)
      alpha_optimizer = optax.adam(learning_rate=3e-4)
      alpha_optimizer_state = alpha_optimizer.init(log_alpha)
    else:
      if target_entropy:
        raise ValueError('target_entropy should not be set when '
                         'entropy_coefficient is provided')

    def alpha_loss(log_alpha: jnp.ndarray,
                   policy_params: networks_lib.Params,
                   transitions: types.Transition,
                   key: networks_lib.PRNGKey) -> jnp.ndarray:
      """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
      dist_params = networks.policy_network.apply(
          policy_params, transitions.observation)
      action = networks.sample(dist_params, key)
      log_prob = networks.log_prob(dist_params, action)
      alpha = jnp.exp(log_alpha)
      alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
      return jnp.mean(alpha_loss)

    def critic_loss(q_params: networks_lib.Params,
                    policy_params: networks_lib.Params,
                    target_q_params: networks_lib.Params,
                    alpha: jnp.ndarray,
                    transitions: types.Transition,
                    key: networks_lib.PRNGKey) -> jnp.ndarray:
      q_old_action = networks.q_network.apply(
          q_params, transitions.observation, transitions.action)
      next_dist_params = networks.policy_network.apply(
          policy_params, transitions.next_observation)
      next_action = networks.sample(next_dist_params, key)
      next_log_prob = networks.log_prob(next_dist_params, next_action)
      next_q = networks.q_network.apply(
          target_q_params, transitions.next_observation, next_action)
      next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
      target_q = jax.lax.stop_gradient(transitions.reward * reward_scale +
                                       transitions.discount * discount * next_v)
      q_error = q_old_action - jnp.expand_dims(target_q, -1)
      q_loss = 0.5 * jnp.mean(jnp.square(q_error))
      return q_loss

    def actor_loss(policy_params: networks_lib.Params,
                   q_params: networks_lib.Params,
                   alpha: jnp.ndarray,
                   transitions: types.Transition,
                   key: networks_lib.PRNGKey) -> jnp.ndarray:
      dist_params = networks.policy_network.apply(
          policy_params, transitions.observation)
      action = networks.sample(dist_params, key)
      log_prob = networks.log_prob(dist_params, action)
      q_action = networks.q_network.apply(
          q_params, transitions.observation, action)
      min_q = jnp.min(q_action, axis=-1)
      actor_loss = alpha * log_prob - min_q
      return jnp.mean(actor_loss)

    alpha_grad = jax.value_and_grad(alpha_loss)
    critic_grad = jax.value_and_grad(critic_loss)
    actor_grad = jax.value_and_grad(actor_loss)

    def update_step(
        state: TrainingState,
        transitions: types.Transition,
    ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:

      key, key_alpha, key_critic, key_actor = jax.random.split(state.key, 4)
      if adaptive_entropy_coefficient:
        alpha_loss, alpha_grads = alpha_grad(state.alpha_params,
                                             state.policy_params, transitions,
                                             key_alpha)
        alpha = jnp.exp(state.alpha_params)
      else:
        alpha = entropy_coefficient
      critic_loss, critic_grads = critic_grad(state.q_params,
                                              state.policy_params,
                                              state.target_q_params, alpha,
                                              transitions, key_critic)
      actor_loss, actor_grads = actor_grad(state.policy_params, state.q_params,
                                           alpha, transitions, key_actor)

      # Apply policy gradients
      actor_update, policy_optimizer_state = policy_optimizer.update(
          actor_grads, state.policy_optimizer_state)
      policy_params = optax.apply_updates(state.policy_params, actor_update)

      # Apply critic gradients
      critic_update, q_optimizer_state = q_optimizer.update(
          critic_grads, state.q_optimizer_state)
      q_params = optax.apply_updates(state.q_params, critic_update)

      new_target_q_params = jax.tree_multimap(
          lambda x, y: x * (1 - tau) + y * tau, state.target_q_params, q_params)

      metrics = {
          'critic_loss': critic_loss,
          'actor_loss': actor_loss,
      }

      new_state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          q_optimizer_state=q_optimizer_state,
          policy_params=policy_params,
          q_params=q_params,
          target_q_params=new_target_q_params,
          key=key,
      )
      if adaptive_entropy_coefficient:
        # Apply alpha gradients
        alpha_update, alpha_optimizer_state = alpha_optimizer.update(
            alpha_grads, state.alpha_optimizer_state)
        alpha_params = optax.apply_updates(state.alpha_params, alpha_update)
        metrics.update({
            'alpha_loss': alpha_loss,
            'alpha': jnp.exp(alpha_params),
        })
        new_state = new_state._replace(
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params)

      metrics['observations_mean'] = jnp.mean(
          utils.batch_concat(
              jax.tree_map(lambda x: jnp.abs(jnp.mean(x, axis=0)),
                           transitions.observation)))
      metrics['observations_std'] = jnp.mean(
          utils.batch_concat(
              jax.tree_map(lambda x: jnp.std(x, axis=0),
                           transitions.observation)))
      metrics['next_observations_mean'] = jnp.mean(
          utils.batch_concat(
              jax.tree_map(lambda x: jnp.abs(jnp.mean(x, axis=0)),
                           transitions.next_observation)))
      metrics['next_observations_std'] = jnp.mean(
          utils.batch_concat(
              jax.tree_map(lambda x: jnp.std(x, axis=0),
                           transitions.next_observation)))

      metrics['rewards_mean'] = jnp.mean(
          jnp.abs(jnp.mean(transitions.reward, axis=0)))
      metrics['rewards_std'] = jnp.std(transitions.reward, axis=0)

      return new_state, metrics

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        'learner',
        asynchronous=True,
        serialize_fn=utils.fetch_devicearray,
        steps_key=self._counter.get_steps_key())

    # Iterator on demonstration transitions.
    self._iterator = iterator

    update_step = utils.process_multiple_batches(update_step,
                                                 num_sgd_steps_per_step)
    # Use the JIT compiler.
    self._update_step = jax.jit(update_step)

    def make_initial_state(key: networks_lib.PRNGKey) -> TrainingState:
      """Initialises the training state (parameters and optimiser state)."""
      key_policy, key_q, key = jax.random.split(key, 3)

      policy_params = networks.policy_network.init(key_policy)
      policy_optimizer_state = policy_optimizer.init(policy_params)

      q_params = networks.q_network.init(key_q)
      q_optimizer_state = q_optimizer.init(q_params)

      state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          q_optimizer_state=q_optimizer_state,
          policy_params=policy_params,
          q_params=q_params,
          target_q_params=q_params,
          key=key)

      if adaptive_entropy_coefficient:
        state = state._replace(alpha_optimizer_state=alpha_optimizer_state,
                               alpha_params=log_alpha)
      return state

    # Create initial state.
    self._state = make_initial_state(rng)

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

  def step(self):
    sample = next(self._iterator)
    transitions = types.Transition(*sample.data)

    self._state, metrics = self._update_step(self._state, transitions)

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Increment counts and record the current time
    counts = self._counter.increment(steps=1, walltime=elapsed_time)

    # Attempts to write the logs.
    self._logger.write({**metrics, **counts})

  def get_variables(self, names: List[str]) -> List[Any]:
    variables = {
        'policy': self._state.policy_params,
        'critic': self._state.q_params,
    }
    return [variables[name] for name in names]

  def save(self) -> TrainingState:
    return self._state

  def restore(self, state: TrainingState):
    self._state = state
