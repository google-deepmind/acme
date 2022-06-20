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

"""CQL learner implementation."""

import time
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple
import acme
from acme import types
from acme.agents.jax.cql.networks import apply_and_sample_n
from acme.agents.jax.cql.networks import CQLNetworks
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import optax


_CQL_COEFFICIENT_MAX_VALUE = 1E6
_CQL_GRAD_CLIPPING_VALUE = 40


class TrainingState(NamedTuple):
  """Contains training state for the learner."""
  policy_optimizer_state: optax.OptState
  critic_optimizer_state: optax.OptState
  policy_params: networks_lib.Params
  critic_params: networks_lib.Params
  target_critic_params: networks_lib.Params
  key: networks_lib.PRNGKey

  # Optimizer and value of the alpha parameter from SAC (entropy temperature).
  # These fields are only used with an adaptive coefficient (when
  # fixed_entropy_coefficeint is None in the CQLLearner)
  alpha_optimizer_state: Optional[optax.OptState] = None
  log_sac_alpha: Optional[networks_lib.Params] = None

  # Optimizer and value of the alpha parameter from CQL (regularization
  # coefficient).
  # These fields are only used with an adaptive coefficient (when
  # fixed_cql_coefficiennt is None in the CQLLearner)
  cql_optimizer_state: Optional[optax.OptState] = None
  log_cql_alpha: Optional[networks_lib.Params] = None

  steps: int = 0


class CQLLearner(acme.Learner):
  """CQL learner.

  Learning component of the Conservative Q-Learning algorithm from
  [Kumar et al., 2020] https://arxiv.org/abs/2006.04779.
  """

  _state: TrainingState

  def __init__(self,
               batch_size: int,
               networks: CQLNetworks,
               random_key: networks_lib.PRNGKey,
               demonstrations: Iterator[types.Transition],
               policy_optimizer: optax.GradientTransformation,
               critic_optimizer: optax.GradientTransformation,
               tau: float = 0.005,
               fixed_cql_coefficient: Optional[float] = None,
               cql_lagrange_threshold: Optional[float] = None,
               cql_num_samples: int = 10,
               num_sgd_steps_per_step: int = 1,
               reward_scale: float = 1.0,
               discount: float = 0.99,
               fixed_entropy_coefficient: Optional[float] = None,
               target_entropy: Optional[float] = 0,
               num_bc_iters: int = 50_000,
               counter: Optional[counting.Counter] = None,
               logger: Optional[loggers.Logger] = None):
    """Initializes the CQL learner.

    Args:
      batch_size: bath size.
      networks: CQL networks.
      random_key: a key for random number generation.
      demonstrations: an iterator over training data.
      policy_optimizer: the policy optimizer.
      critic_optimizer: the Q-function optimizer.
      tau: target smoothing coefficient.
      fixed_cql_coefficient: the value for cql coefficient. If None, an adaptive
        coefficient will be used.
      cql_lagrange_threshold: a threshold that controls the adaptive loss for
        the cql coefficient.
      cql_num_samples: number of samples used to compute logsumexp(Q) via
        importance sampling.
      num_sgd_steps_per_step: how many gradient updated to perform per batch.
        batch is split into this many smaller batches, thus should be a multiple
        of num_sgd_steps_per_step
      reward_scale: reward scale.
      discount: discount to use for TD updates.
      fixed_entropy_coefficient: coefficient applied to the entropy bonus. If
        None, an adaptative coefficient will be used.
      target_entropy: Target entropy when using adapdative entropy bonus.
      num_bc_iters: Number of BC steps for actor initialization.
      counter: counter object used to keep track of steps.
      logger: logger object to be used by learner.
    """
    self._num_bc_iters = num_bc_iters
    adaptive_entropy_coefficient = fixed_entropy_coefficient is None
    action_spec = networks.environment_specs.actions
    if adaptive_entropy_coefficient:
      # sac_alpha is the temperature parameter that determines the relative
      # importance of the entropy term versus the reward.
      log_sac_alpha = jnp.asarray(0., dtype=jnp.float32)
      alpha_optimizer = optax.adam(learning_rate=3e-4)
      alpha_optimizer_state = alpha_optimizer.init(log_sac_alpha)
    else:
      if target_entropy:
        raise ValueError('target_entropy should not be set when '
                         'fixed_entropy_coefficient is provided')

    adaptive_cql_coefficient = fixed_cql_coefficient is None
    if adaptive_cql_coefficient:
      log_cql_alpha = jnp.asarray(0., dtype=jnp.float32)
      cql_optimizer = optax.adam(learning_rate=3e-4)
      cql_optimizer_state = cql_optimizer.init(log_cql_alpha)
    else:
      if cql_lagrange_threshold:
        raise ValueError('cql_lagrange_threshold should not be set when '
                         'fixed_cql_coefficient is provided')

    def alpha_loss(log_sac_alpha: jnp.ndarray,
                   policy_params: networks_lib.Params,
                   transitions: types.Transition,
                   key: jnp.ndarray) -> jnp.ndarray:
      """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
      dist_params = networks.policy_network.apply(policy_params,
                                                  transitions.observation)
      action = networks.sample(dist_params, key)
      log_prob = networks.log_prob(dist_params, action)
      sac_alpha = jnp.exp(log_sac_alpha)
      sac_alpha_loss = sac_alpha * jax.lax.stop_gradient(-log_prob -
                                                         target_entropy)
      return jnp.mean(sac_alpha_loss)

    def sac_critic_loss(q_old_action: jnp.ndarray,
                        policy_params: networks_lib.Params,
                        target_critic_params: networks_lib.Params,
                        transitions: types.Transition,
                        key: networks_lib.PRNGKey) -> jnp.ndarray:
      """Computes the SAC part of the loss."""
      next_dist_params = networks.policy_network.apply(
          policy_params, transitions.next_observation)
      next_action = networks.sample(next_dist_params, key)
      next_q = networks.critic_network.apply(target_critic_params,
                                             transitions.next_observation,
                                             next_action)
      next_v = jnp.min(next_q, axis=-1)
      target_q = jax.lax.stop_gradient(transitions.reward * reward_scale +
                                       transitions.discount * discount * next_v)
      return jnp.mean(jnp.square(q_old_action - jnp.expand_dims(target_q, -1)))

    def batched_critic(actions: jnp.ndarray, critic_params: networks_lib.Params,
                       observation: jnp.ndarray) -> jnp.ndarray:
      """Applies the critic network to a batch of sampled actions."""
      actions = jax.lax.stop_gradient(actions)
      tiled_actions = jnp.reshape(actions, (batch_size * cql_num_samples, -1))
      tiled_states = jnp.tile(observation, [cql_num_samples, 1])
      tiled_q = networks.critic_network.apply(critic_params, tiled_states,
                                              tiled_actions)
      return jnp.reshape(tiled_q, (cql_num_samples, batch_size, -1))

    def cql_critic_loss(q_old_action: jnp.ndarray,
                        critic_params: networks_lib.Params,
                        policy_params: networks_lib.Params,
                        transitions: types.Transition,
                        key: networks_lib.PRNGKey) -> jnp.ndarray:
      """Computes the CQL part of the loss."""
      # The CQL part of the loss is
      #     logsumexp(Q(s,·)) - Q(s,a),
      # where s is the currrent state, and a the action in the dataset (so
      # Q(s,a) is simply q_old_action.
      # We need to estimate logsumexp(Q). This is done with importance sampling
      # (IS). This function implements the unlabeled equation page 29, Appx. F,
      # in https://arxiv.org/abs/2006.04779.
      # Here, IS is done with the uniform distribution and the policy in the
      # current state s. In their implementation, the authors also add the
      # policy in the transiting state s':
      # https://github.com/aviralkumar2907/CQL/blob/master/d4rl/rlkit/torch/sac/cql.py,
      # (l. 233-236).

      key_policy, key_policy_next, key_uniform = jax.random.split(key, 3)

      def sampled_q(obs, key):
        actions, log_probs = apply_and_sample_n(
            key, networks, policy_params, obs, cql_num_samples)
        return batched_critic(actions, critic_params,
                              transitions.observation) - jax.lax.stop_gradient(
                                  jnp.expand_dims(log_probs, -1))

      # Sample wrt policy in s
      sampled_q_from_policy = sampled_q(transitions.observation, key_policy)

      # Sample wrt policy in s'
      sampled_q_from_policy_next = sampled_q(transitions.next_observation,
                                             key_policy_next)

      # Sample wrt uniform
      actions_uniform = jax.random.uniform(
          key_uniform, (cql_num_samples, batch_size) + action_spec.shape,
          minval=action_spec.minimum, maxval=action_spec.maximum)
      log_prob_uniform = -jnp.sum(
          jnp.log(action_spec.maximum - action_spec.minimum))
      sampled_q_from_uniform = (
          batched_critic(actions_uniform, critic_params,
                         transitions.observation) - log_prob_uniform)

      # Combine the samplings
      combined = jnp.concatenate(
          (sampled_q_from_uniform, sampled_q_from_policy,
           sampled_q_from_policy_next),
          axis=0)
      lse_q = jax.nn.logsumexp(combined, axis=0, b=1. / (3 * cql_num_samples))

      return jnp.mean(lse_q - q_old_action)

    def critic_loss(critic_params: networks_lib.Params,
                    policy_params: networks_lib.Params,
                    target_critic_params: networks_lib.Params,
                    cql_alpha: jnp.ndarray, transitions: types.Transition,
                    key: networks_lib.PRNGKey) -> jnp.ndarray:
      """Computes the full critic loss."""
      key_cql, key_sac = jax.random.split(key, 2)
      q_old_action = networks.critic_network.apply(critic_params,
                                                   transitions.observation,
                                                   transitions.action)
      cql_loss = cql_critic_loss(q_old_action, critic_params, policy_params,
                                 transitions, key_cql)
      sac_loss = sac_critic_loss(q_old_action, policy_params,
                                 target_critic_params, transitions, key_sac)
      return cql_alpha * cql_loss + sac_loss

    def cql_lagrange_loss(log_cql_alpha: jnp.ndarray,
                          critic_params: networks_lib.Params,
                          policy_params: networks_lib.Params,
                          transitions: types.Transition,
                          key: jnp.ndarray) -> jnp.ndarray:
      """Computes the loss that optimizes the cql coefficient."""
      cql_alpha = jnp.exp(log_cql_alpha)
      q_old_action = networks.critic_network.apply(critic_params,
                                                   transitions.observation,
                                                   transitions.action)
      return -cql_alpha * (
          cql_critic_loss(q_old_action, critic_params, policy_params,
                          transitions, key) - cql_lagrange_threshold)

    def actor_loss(policy_params: networks_lib.Params,
                   critic_params: networks_lib.Params, sac_alpha: jnp.ndarray,
                   transitions: types.Transition, key: jnp.ndarray,
                   in_initial_bc_iters: bool) -> jnp.ndarray:
      """Computes the loss for the policy."""
      dist_params = networks.policy_network.apply(policy_params,
                                                  transitions.observation)
      if in_initial_bc_iters:
        log_prob = networks.log_prob(dist_params, transitions.action)
        actor_loss = -jnp.mean(log_prob)
      else:
        action = networks.sample(dist_params, key)
        log_prob = networks.log_prob(dist_params, action)
        q_action = networks.critic_network.apply(critic_params,
                                                 transitions.observation,
                                                 action)
        min_q = jnp.min(q_action, axis=-1)
        actor_loss = jnp.mean(sac_alpha * log_prob - min_q)
      return actor_loss

    alpha_grad = jax.value_and_grad(alpha_loss)
    cql_lagrange_grad = jax.value_and_grad(cql_lagrange_loss)
    critic_grad = jax.value_and_grad(critic_loss)
    actor_grad = jax.value_and_grad(actor_loss)

    def update_step(
        state: TrainingState,
        rb_transitions: types.Transition,
        in_initial_bc_iters: bool,
    ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:

      key, key_alpha, key_critic, key_actor = jax.random.split(state.key, 4)

      if adaptive_entropy_coefficient:
        alpha_loss, alpha_grads = alpha_grad(state.log_sac_alpha,
                                             state.policy_params,
                                             rb_transitions, key_alpha)
        sac_alpha = jnp.exp(state.log_sac_alpha)
      else:
        sac_alpha = fixed_entropy_coefficient

      if adaptive_cql_coefficient:
        cql_lagrange_loss, cql_lagrange_grads = cql_lagrange_grad(
            state.log_cql_alpha, state.critic_params, state.policy_params,
            rb_transitions, key_critic)
        cql_lagrange_grads = jnp.clip(cql_lagrange_grads,
                                      -_CQL_GRAD_CLIPPING_VALUE,
                                      _CQL_GRAD_CLIPPING_VALUE)
        cql_alpha = jnp.exp(state.log_cql_alpha)
        cql_alpha = jnp.clip(
            cql_alpha, a_min=0., a_max=_CQL_COEFFICIENT_MAX_VALUE)
      else:
        cql_alpha = fixed_cql_coefficient

      critic_loss, critic_grads = critic_grad(state.critic_params,
                                              state.policy_params,
                                              state.target_critic_params,
                                              cql_alpha, rb_transitions,
                                              key_critic)
      actor_loss, actor_grads = actor_grad(state.policy_params,
                                           state.critic_params, sac_alpha,
                                           rb_transitions, key_actor,
                                           in_initial_bc_iters)

      # Apply policy gradients
      actor_update, policy_optimizer_state = policy_optimizer.update(
          actor_grads, state.policy_optimizer_state)
      policy_params = optax.apply_updates(state.policy_params, actor_update)

      # Apply critic gradients
      critic_update, critic_optimizer_state = critic_optimizer.update(
          critic_grads, state.critic_optimizer_state)
      critic_params = optax.apply_updates(state.critic_params, critic_update)

      new_target_critic_params = jax.tree_multimap(
          lambda x, y: x * (1 - tau) + y * tau, state.target_critic_params,
          critic_params)

      metrics = {
          'critic_loss': critic_loss,
          'actor_loss': actor_loss,
      }

      new_state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          critic_optimizer_state=critic_optimizer_state,
          policy_params=policy_params,
          critic_params=critic_params,
          target_critic_params=new_target_critic_params,
          key=key,
          alpha_optimizer_state=state.alpha_optimizer_state,
          log_sac_alpha=state.log_sac_alpha,
          steps=state.steps + 1,
      )
      if adaptive_entropy_coefficient and (not in_initial_bc_iters):
        # Apply sac_alpha gradients
        alpha_update, alpha_optimizer_state = alpha_optimizer.update(
            alpha_grads, state.alpha_optimizer_state)
        log_sac_alpha = optax.apply_updates(state.log_sac_alpha, alpha_update)
        metrics.update({
            'alpha_loss': alpha_loss,
            'sac_alpha': jnp.exp(log_sac_alpha),
        })
        new_state = new_state._replace(
            alpha_optimizer_state=alpha_optimizer_state,
            log_sac_alpha=log_sac_alpha)
      else:
        metrics['alpha_loss'] = 0.
        metrics['sac_alpha'] = fixed_cql_coefficient

      if adaptive_cql_coefficient:
        # Apply cql coeff gradients
        cql_update, cql_optimizer_state = cql_optimizer.update(
            cql_lagrange_grads, state.cql_optimizer_state)
        log_cql_alpha = optax.apply_updates(state.log_cql_alpha, cql_update)
        metrics.update({
            'cql_lagrange_loss': cql_lagrange_loss,
            'cql_alpha': jnp.exp(log_cql_alpha),
        })
        new_state = new_state._replace(
            cql_optimizer_state=cql_optimizer_state,
            log_cql_alpha=log_cql_alpha)

      return new_state, metrics

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        'learner',
        asynchronous=True,
        serialize_fn=utils.fetch_devicearray,
        steps_key=self._counter.get_steps_key())

    # Iterator on demonstration transitions.
    self._demonstrations = demonstrations

    # Use the JIT compiler.
    update_step_in_initial_bc_iters = utils.process_multiple_batches(
        lambda x, y: update_step(x, y, True), num_sgd_steps_per_step)
    update_step_rest = utils.process_multiple_batches(
        lambda x, y: update_step(x, y, False), num_sgd_steps_per_step)

    self._update_step_in_initial_bc_iters = jax.jit(
        update_step_in_initial_bc_iters)
    self._update_step_rest = jax.jit(update_step_rest)

    # Create initial state.
    key_policy, key_q, training_state_key = jax.random.split(random_key, 3)
    del random_key
    policy_params = networks.policy_network.init(key_policy)
    policy_optimizer_state = policy_optimizer.init(policy_params)
    critic_params = networks.critic_network.init(key_q)
    critic_optimizer_state = critic_optimizer.init(critic_params)

    self._state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        critic_optimizer_state=critic_optimizer_state,
        policy_params=policy_params,
        critic_params=critic_params,
        target_critic_params=critic_params,
        key=training_state_key,
        steps=0)

    if adaptive_entropy_coefficient:
      self._state = self._state._replace(
          alpha_optimizer_state=alpha_optimizer_state,
          log_sac_alpha=log_sac_alpha)
    if adaptive_cql_coefficient:
      self._state = self._state._replace(
          cql_optimizer_state=cql_optimizer_state, log_cql_alpha=log_cql_alpha)

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

  def step(self):
    # Get data from replay (dropping extras if any). Note there is no
    # extra data here because we do not insert any into Reverb.
    transitions = next(self._demonstrations)

    counts = self._counter.get_counts()
    if 'learner_steps' not in counts:
      cur_step = 0
    else:
      cur_step = counts['learner_steps']
    in_initial_bc_iters = cur_step < self._num_bc_iters

    if in_initial_bc_iters:
      self._state, metrics = self._update_step_in_initial_bc_iters(
          self._state, transitions)
    else:
      self._state, metrics = self._update_step_rest(self._state, transitions)

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
    }
    return [variables[name] for name in names]

  def save(self) -> TrainingState:
    return self._state

  def restore(self, state: TrainingState):
    self._state = state
