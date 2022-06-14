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

"""CRR learner implementation."""

import time
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import acme
from acme import types
from acme.agents.jax.crr.losses import PolicyLossCoeff
from acme.agents.jax.crr.networks import CRRNetworks
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import optax


class TrainingState(NamedTuple):
  """Contains training state for the learner."""
  policy_params: networks_lib.Params
  target_policy_params: networks_lib.Params
  critic_params: networks_lib.Params
  target_critic_params: networks_lib.Params
  policy_opt_state: optax.OptState
  critic_opt_state: optax.OptState
  steps: int
  key: networks_lib.PRNGKey


class CRRLearner(acme.Learner):
  """Critic Regularized Regression (CRR) learner.

  This is the learning component of a CRR agent as described in
  https://arxiv.org/abs/2006.15134.
  """

  _state: TrainingState

  def __init__(self,
               networks: CRRNetworks,
               random_key: networks_lib.PRNGKey,
               discount: float,
               target_update_period: int,
               policy_loss_coeff_fn: PolicyLossCoeff,
               iterator: Iterator[types.Transition],
               policy_optimizer: optax.GradientTransformation,
               critic_optimizer: optax.GradientTransformation,
               counter: Optional[counting.Counter] = None,
               logger: Optional[loggers.Logger] = None,
               grad_updates_per_batch: int = 1,
               use_sarsa_target: bool = False):
    """Initializes the CRR learner.

    Args:
      networks: CRR networks.
      random_key: a key for random number generation.
      discount: discount to use for TD updates.
      target_update_period: period to update target's parameters.
      policy_loss_coeff_fn: set the loss function for the policy.
      iterator: an iterator over training data.
      policy_optimizer: the policy optimizer.
      critic_optimizer: the Q-function optimizer.
      counter: counter object used to keep track of steps.
      logger: logger object to be used by learner.
      grad_updates_per_batch: how many gradient updates given a sampled batch.
      use_sarsa_target: compute on-policy target using iterator's actions rather
        than sampled actions.
        Useful for 1-step offline RL (https://arxiv.org/pdf/2106.08909.pdf).
        When set to `True`, `target_policy_params` are unused.
    """

    critic_network = networks.critic_network
    policy_network = networks.policy_network

    def policy_loss(
        policy_params: networks_lib.Params,
        critic_params: networks_lib.Params,
        transition: types.Transition,
        key: networks_lib.PRNGKey,
    ) -> jnp.ndarray:
      # Compute the loss coefficients.
      coeff = policy_loss_coeff_fn(networks, policy_params, critic_params,
                                   transition, key)
      coeff = jax.lax.stop_gradient(coeff)
      # Return the weighted loss.
      dist_params = policy_network.apply(policy_params, transition.observation)
      logp_action = networks.log_prob(dist_params, transition.action)
      return -jnp.mean(logp_action * coeff)

    def critic_loss(
        critic_params: networks_lib.Params,
        target_policy_params: networks_lib.Params,
        target_critic_params: networks_lib.Params,
        transition: types.Transition,
        key: networks_lib.PRNGKey,
    ):
      # Sample the next action.
      if use_sarsa_target:
        # TODO(b/222674779): use N-steps Trajectories to get the next actions.
        assert 'next_action' in transition.extras, (
            'next actions should be given as extras for one step RL.')
        next_action = transition.extras['next_action']
      else:
        next_dist_params = policy_network.apply(target_policy_params,
                                                transition.next_observation)
        next_action = networks.sample(next_dist_params, key)
      # Calculate the value of the next state and action.
      next_q = critic_network.apply(target_critic_params,
                                    transition.next_observation, next_action)
      target_q = transition.reward + transition.discount * discount * next_q
      target_q = jax.lax.stop_gradient(target_q)

      q = critic_network.apply(critic_params, transition.observation,
                               transition.action)
      q_error = q - target_q
      # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
      # TODO(sertan): Replace with a distributional critic. CRR paper states
      # that this may perform better.
      return 0.5 * jnp.mean(jnp.square(q_error))

    policy_loss_and_grad = jax.value_and_grad(policy_loss)
    critic_loss_and_grad = jax.value_and_grad(critic_loss)

    def sgd_step(
        state: TrainingState,
        transitions: types.Transition,
    ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:

      key, key_policy, key_critic = jax.random.split(state.key, 3)

      # Compute losses and their gradients.
      policy_loss_value, policy_gradients = policy_loss_and_grad(
          state.policy_params, state.critic_params, transitions, key_policy)
      critic_loss_value, critic_gradients = critic_loss_and_grad(
          state.critic_params, state.target_policy_params,
          state.target_critic_params, transitions, key_critic)

      # Get optimizer updates and state.
      policy_updates, policy_opt_state = policy_optimizer.update(
          policy_gradients, state.policy_opt_state)
      critic_updates, critic_opt_state = critic_optimizer.update(
          critic_gradients, state.critic_opt_state)

      # Apply optimizer updates to parameters.
      policy_params = optax.apply_updates(state.policy_params, policy_updates)
      critic_params = optax.apply_updates(state.critic_params, critic_updates)

      steps = state.steps + 1

      # Periodically update target networks.
      target_policy_params, target_critic_params = optax.periodic_update(
          (policy_params, critic_params),
          (state.target_policy_params, state.target_critic_params), steps,
          target_update_period)

      new_state = TrainingState(
          policy_params=policy_params,
          target_policy_params=target_policy_params,
          critic_params=critic_params,
          target_critic_params=target_critic_params,
          policy_opt_state=policy_opt_state,
          critic_opt_state=critic_opt_state,
          steps=steps,
          key=key,
      )

      metrics = {
          'policy_loss': policy_loss_value,
          'critic_loss': critic_loss_value,
      }

      return new_state, metrics

    sgd_step = utils.process_multiple_batches(sgd_step, grad_updates_per_batch)
    self._sgd_step = jax.jit(sgd_step)

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        'learner',
        asynchronous=True,
        serialize_fn=utils.fetch_devicearray,
        steps_key=self._counter.get_steps_key())

    # Create prefetching dataset iterator.
    self._iterator = iterator

    # Create the network parameters and copy into the target network parameters.
    key, key_policy, key_critic = jax.random.split(random_key, 3)
    initial_policy_params = policy_network.init(key_policy)
    initial_critic_params = critic_network.init(key_critic)
    initial_target_policy_params = initial_policy_params
    initial_target_critic_params = initial_critic_params

    # Initialize optimizers.
    initial_policy_opt_state = policy_optimizer.init(initial_policy_params)
    initial_critic_opt_state = critic_optimizer.init(initial_critic_params)

    # Create initial state.
    self._state = TrainingState(
        policy_params=initial_policy_params,
        target_policy_params=initial_target_policy_params,
        critic_params=initial_critic_params,
        target_critic_params=initial_target_critic_params,
        policy_opt_state=initial_policy_opt_state,
        critic_opt_state=initial_critic_opt_state,
        steps=0,
        key=key,
    )

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

  def step(self):
    transitions = next(self._iterator)

    self._state, metrics = self._sgd_step(self._state, transitions)

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Increment counts and record the current time
    counts = self._counter.increment(steps=1, walltime=elapsed_time)

    # Attempts to write the logs.
    self._logger.write({**metrics, **counts})

  def get_variables(self, names: List[str]) -> List[networks_lib.Params]:
    # We only expose the variables for the learned policy and critic. The target
    # policy and critic are internal details.
    variables = {
        'policy': self._state.target_policy_params,
        'critic': self._state.target_critic_params,
    }
    return [variables[name] for name in names]

  def save(self) -> TrainingState:
    return self._state

  def restore(self, state: TrainingState):
    self._state = state
