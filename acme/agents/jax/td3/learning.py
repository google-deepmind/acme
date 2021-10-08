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

"""TD3 learner implementation."""

import time
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import acme
from acme import types
from acme.agents.jax.td3 import networks as td3_networks
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import optax
import reverb
import rlax


class TrainingState(NamedTuple):
  """Contains training state for the learner."""
  policy_params: networks_lib.Params
  target_policy_params: networks_lib.Params
  critic_params: networks_lib.Params
  target_critic_params: networks_lib.Params
  twin_critic_params: networks_lib.Params
  target_twin_critic_params: networks_lib.Params
  policy_opt_state: optax.OptState
  critic_opt_state: optax.OptState
  twin_critic_opt_state: optax.OptState
  steps: int
  random_key: networks_lib.PRNGKey


class TD3Learner(acme.Learner):
  """TD3 learner."""

  _state: TrainingState

  def __init__(self,
               networks: td3_networks.TD3Networks,
               random_key: networks_lib.PRNGKey,
               discount: float,
               iterator: Iterator[reverb.ReplaySample],
               policy_optimizer: optax.GradientTransformation,
               critic_optimizer: optax.GradientTransformation,
               twin_critic_optimizer: optax.GradientTransformation,
               delay: int = 2,
               target_sigma: float = 0.2,
               noise_clip: float = 0.5,
               tau: float = 0.005,
               counter: Optional[counting.Counter] = None,
               logger: Optional[loggers.Logger] = None,
               num_sgd_steps_per_step: int = 1):
    """Initializes the TD3 learner.

    Args:
      networks: TD3 networks.
      random_key: a key for random number generation.
      discount: discount to use for TD updates
      iterator: an iterator over training data.
      policy_optimizer: the policy optimizer.
      critic_optimizer: the Q-function optimizer.
      twin_critic_optimizer: the twin Q-function optimizer.
      delay: ratio of policy updates for critic updates (see TD3),
        delay=2 means 2 updates of the critic for 1 policy update.
      target_sigma: std of zero mean Gaussian added to the action of
        the next_state, for critic evaluation (reducing overestimation bias).
      noise_clip: hard constraint on target noise.
      tau: target parameters smoothing coefficient.
      counter: counter object used to keep track of steps.
      logger: logger object to be used by learner.
      num_sgd_steps_per_step: number of sgd steps to perform per learner 'step'.
    """

    def policy_loss(
        policy_params: networks_lib.Params,
        critic_params: networks_lib.Params,
        observation: types.NestedArray,
    ) -> jnp.ndarray:
      # Computes the discrete policy gradient loss.
      action = networks.policy_network.apply(policy_params, observation)
      grad_critic = jax.vmap(
          jax.grad(networks.critic_network.apply, argnums=2),
          in_axes=(None, 0, 0))
      dq_da = grad_critic(critic_params, observation, action)
      batch_dpg_learning = jax.vmap(rlax.dpg_loss, in_axes=(0, 0))
      loss = batch_dpg_learning(action, dq_da)
      return jnp.mean(loss)

    def critic_loss(
        critic_params: networks_lib.Params,
        state: TrainingState,
        transition: types.Transition,
        random_key: jnp.ndarray,
    ):
      # Computes the critic loss.
      q_tm1 = networks.critic_network.apply(
          critic_params, transition.observation, transition.action)

      action = networks.policy_network.apply(state.target_policy_params,
                                             transition.next_observation)
      action_with_noise = networks.add_policy_noise(action, random_key,
                                                    target_sigma, noise_clip)

      q_t = networks.critic_network.apply(
          state.target_critic_params,
          transition.next_observation,
          action_with_noise)
      twin_q_t = networks.twin_critic_network.apply(
          state.target_twin_critic_params,
          transition.next_observation,
          action_with_noise)

      q_t = jnp.minimum(q_t, twin_q_t)

      target_q_tm1 = transition.reward + discount * transition.discount * q_t
      td_error = jax.lax.stop_gradient(target_q_tm1) - q_tm1

      return jnp.mean(jnp.square(td_error))

    def update_step(
        state: TrainingState,
        transitions: types.Transition,
    ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:

      random_key, key_critic, key_twin = jax.random.split(state.random_key, 3)
      polyak_averaging = lambda x, y: x * (1 - tau) + y * tau

      # Updates on the critic: compute the gradients, and update using
      # Polyak averaging.
      critic_loss_and_grad = jax.value_and_grad(critic_loss)
      critic_loss_value, critic_gradients = critic_loss_and_grad(
          state.critic_params, state, transitions, key_critic)
      critic_updates, critic_opt_state = critic_optimizer.update(
          critic_gradients, state.critic_opt_state)
      critic_params = optax.apply_updates(state.critic_params, critic_updates)
      target_critic_params = jax.tree_multimap(
          polyak_averaging, state.target_critic_params, critic_params)

      # Updates on the twin critic: compute the gradients, and update using
      # Polyak averaging.
      twin_critic_loss_value, twin_critic_gradients = critic_loss_and_grad(
          state.twin_critic_params, state, transitions, key_twin)
      twin_critic_updates, twin_critic_opt_state = twin_critic_optimizer.update(
          twin_critic_gradients, state.twin_critic_opt_state)
      twin_critic_params = optax.apply_updates(state.twin_critic_params,
                                               twin_critic_updates)
      target_twin_critic_params = jax.tree_multimap(
          polyak_averaging, state.target_twin_critic_params, twin_critic_params)

      # Updates on the policy: compute the gradients, and update using
      # Polyak averaging (if delay enabled, the update might not be applied).
      policy_loss_and_grad = jax.value_and_grad(policy_loss)
      policy_loss_value, policy_gradients = policy_loss_and_grad(
          state.policy_params, state.critic_params,
          transitions.next_observation)
      def update_policy_step():
        policy_updates, policy_opt_state = policy_optimizer.update(
            policy_gradients, state.policy_opt_state)
        policy_params = optax.apply_updates(state.policy_params, policy_updates)
        target_policy_params = jax.tree_multimap(
            polyak_averaging, state.target_policy_params, policy_params)
        return policy_params, target_policy_params, policy_opt_state

      # The update on the policy is applied every `delay` steps.
      current_policy_state = (state.policy_params, state.target_policy_params,
                              state.policy_opt_state)
      policy_params, target_policy_params, policy_opt_state = jax.lax.cond(
          state.steps % delay == 0,
          lambda _: update_policy_step(),
          lambda _: current_policy_state,
          operand=None)

      steps = state.steps + 1

      new_state = TrainingState(
          policy_params=policy_params,
          critic_params=critic_params,
          twin_critic_params=twin_critic_params,
          target_policy_params=target_policy_params,
          target_critic_params=target_critic_params,
          target_twin_critic_params=target_twin_critic_params,
          policy_opt_state=policy_opt_state,
          critic_opt_state=critic_opt_state,
          twin_critic_opt_state=twin_critic_opt_state,
          steps=steps,
          random_key=random_key,
      )

      metrics = {
          'policy_loss': policy_loss_value,
          'critic_loss': critic_loss_value,
          'twin_critic_loss': twin_critic_loss_value,
      }

      return new_state, metrics

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        'learner', asynchronous=True, serialize_fn=utils.fetch_devicearray)

    # Create prefetching dataset iterator.
    self._iterator = iterator

    # Faster sgd step
    update_step = utils.process_multiple_batches(update_step,
                                                 num_sgd_steps_per_step)
    # Use the JIT compiler.
    self._update_step = jax.jit(update_step)

    (key_init_policy, key_init_twin, key_init_target, key_state
     ) = jax.random.split(random_key, 4)
    # Create the network parameters and copy into the target network parameters.
    initial_policy_params = networks.policy_network.init(key_init_policy)
    initial_critic_params = networks.critic_network.init(key_init_twin)
    initial_twin_critic_params = networks.twin_critic_network.init(
        key_init_target)

    initial_target_policy_params = initial_policy_params
    initial_target_critic_params = initial_critic_params
    initial_target_twin_critic_params = initial_twin_critic_params

    # Initialize optimizers.
    initial_policy_opt_state = policy_optimizer.init(initial_policy_params)
    initial_critic_opt_state = critic_optimizer.init(initial_critic_params)
    initial_twin_critic_opt_state = twin_critic_optimizer.init(
        initial_twin_critic_params)

    # Create initial state.
    self._state = TrainingState(
        policy_params=initial_policy_params,
        target_policy_params=initial_target_policy_params,
        critic_params=initial_critic_params,
        twin_critic_params=initial_twin_critic_params,
        target_critic_params=initial_target_critic_params,
        target_twin_critic_params=initial_target_twin_critic_params,
        policy_opt_state=initial_policy_opt_state,
        critic_opt_state=initial_critic_opt_state,
        twin_critic_opt_state=initial_twin_critic_opt_state,
        steps=0,
        random_key=key_state
    )

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

  def step(self):
    # Get data from replay (dropping extras if any). Note there is no
    # extra data here because we do not insert any into Reverb.
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

  def get_variables(self, names: List[str]) -> List[networks_lib.Params]:
    variables = {
        'policy': self._state.policy_params,
        'critic': self._state.critic_params,
        'twin_critic': self._state.twin_critic_params,
    }
    return [variables[name] for name in names]

  def save(self) -> TrainingState:
    return self._state

  def restore(self, state: TrainingState):
    self._state = state
