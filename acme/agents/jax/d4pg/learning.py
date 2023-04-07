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

"""D4PG learner implementation."""

import time
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
import reverb
import rlax

import acme
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting, loggers

_PMAP_AXIS_NAME = "data"


class TrainingState(NamedTuple):
    """Contains training state for the learner."""

    policy_params: networks_lib.Params
    target_policy_params: networks_lib.Params
    critic_params: networks_lib.Params
    target_critic_params: networks_lib.Params
    policy_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    steps: int


class D4PGLearner(acme.Learner):
    """D4PG learner.

  This is the learning component of a D4PG agent. IE it takes a dataset as input
  and implements update functionality to learn from this dataset.
  """

    _state: TrainingState

    def __init__(
        self,
        policy_network: networks_lib.FeedForwardNetwork,
        critic_network: networks_lib.FeedForwardNetwork,
        random_key: networks_lib.PRNGKey,
        discount: float,
        target_update_period: int,
        iterator: Iterator[reverb.ReplaySample],
        policy_optimizer: Optional[optax.GradientTransformation] = None,
        critic_optimizer: Optional[optax.GradientTransformation] = None,
        clipping: bool = True,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        jit: bool = True,
        num_sgd_steps_per_step: int = 1,
    ):
        def critic_mean(
            critic_params: networks_lib.Params,
            observation: types.NestedArray,
            action: types.NestedArray,
        ) -> jnp.ndarray:
            # We add batch dimension to make sure batch concat in critic_network
            # works correctly.
            observation = utils.add_batch_dim(observation)
            action = utils.add_batch_dim(action)
            # Computes the mean action-value estimate.
            logits, atoms = critic_network.apply(critic_params, observation, action)
            logits = utils.squeeze_batch_dim(logits)
            probabilities = jax.nn.softmax(logits)
            return jnp.sum(probabilities * atoms, axis=-1)

        def policy_loss(
            policy_params: networks_lib.Params,
            critic_params: networks_lib.Params,
            o_t: types.NestedArray,
        ) -> jnp.ndarray:
            # Computes the discrete policy gradient loss.
            dpg_a_t = policy_network.apply(policy_params, o_t)
            grad_critic = jax.vmap(
                jax.grad(critic_mean, argnums=2), in_axes=(None, 0, 0)
            )
            dq_da = grad_critic(critic_params, o_t, dpg_a_t)
            dqda_clipping = 1.0 if clipping else None
            batch_dpg_learning = jax.vmap(rlax.dpg_loss, in_axes=(0, 0, None))
            loss = batch_dpg_learning(dpg_a_t, dq_da, dqda_clipping)
            return jnp.mean(loss)

        def critic_loss(
            critic_params: networks_lib.Params,
            state: TrainingState,
            transition: types.Transition,
        ):
            # Computes the distributional critic loss.
            q_tm1, atoms_tm1 = critic_network.apply(
                critic_params, transition.observation, transition.action
            )
            a = policy_network.apply(
                state.target_policy_params, transition.next_observation
            )
            q_t, atoms_t = critic_network.apply(
                state.target_critic_params, transition.next_observation, a
            )
            batch_td_learning = jax.vmap(
                rlax.categorical_td_learning, in_axes=(None, 0, 0, 0, None, 0)
            )
            loss = batch_td_learning(
                atoms_tm1,
                q_tm1,
                transition.reward,
                discount * transition.discount,
                atoms_t,
                q_t,
            )
            return jnp.mean(loss)

        def sgd_step(
            state: TrainingState, transitions: types.Transition,
        ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:

            # TODO(jaslanides): Use a shared forward pass for efficiency.
            policy_loss_and_grad = jax.value_and_grad(policy_loss)
            critic_loss_and_grad = jax.value_and_grad(critic_loss)

            # Compute losses and their gradients.
            policy_loss_value, policy_gradients = policy_loss_and_grad(
                state.policy_params, state.critic_params, transitions.next_observation
            )
            critic_loss_value, critic_gradients = critic_loss_and_grad(
                state.critic_params, state, transitions
            )

            # Average over all devices.
            policy_loss_value, policy_gradients = jax.lax.pmean(
                (policy_loss_value, policy_gradients), _PMAP_AXIS_NAME
            )
            critic_loss_value, critic_gradients = jax.lax.pmean(
                (critic_loss_value, critic_gradients), _PMAP_AXIS_NAME
            )

            # Get optimizer updates and state.
            (
                policy_updates,
                policy_opt_state,
            ) = policy_optimizer.update(  # pytype: disable=attribute-error
                policy_gradients, state.policy_opt_state
            )
            (
                critic_updates,
                critic_opt_state,
            ) = critic_optimizer.update(  # pytype: disable=attribute-error
                critic_gradients, state.critic_opt_state
            )

            # Apply optimizer updates to parameters.
            policy_params = optax.apply_updates(state.policy_params, policy_updates)
            critic_params = optax.apply_updates(state.critic_params, critic_updates)

            steps = state.steps + 1

            # Periodically update target networks.
            (
                target_policy_params,
                target_critic_params,
            ) = optax.periodic_update(  # pytype: disable=wrong-arg-types  # numpy-scalars
                (policy_params, critic_params),
                (state.target_policy_params, state.target_critic_params),
                steps,
                self._target_update_period,
            )

            new_state = TrainingState(
                policy_params=policy_params,
                critic_params=critic_params,
                target_policy_params=target_policy_params,
                target_critic_params=target_critic_params,
                policy_opt_state=policy_opt_state,
                critic_opt_state=critic_opt_state,
                steps=steps,
            )

            metrics = {
                "policy_loss": policy_loss_value,
                "critic_loss": critic_loss_value,
            }

            return new_state, metrics

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(
            "learner",
            asynchronous=True,
            serialize_fn=utils.fetch_devicearray,
            steps_key=self._counter.get_steps_key(),
        )

        # Necessary to track when to update target networks.
        self._target_update_period = target_update_period

        # Create prefetching dataset iterator.
        self._iterator = iterator

        # Maybe use the JIT compiler.
        sgd_step = utils.process_multiple_batches(sgd_step, num_sgd_steps_per_step)
        self._sgd_step = (
            jax.pmap(sgd_step, _PMAP_AXIS_NAME, devices=jax.devices())
            if jit
            else sgd_step
        )

        # Create the network parameters and copy into the target network parameters.
        key_policy, key_critic = jax.random.split(random_key)
        initial_policy_params = policy_network.init(key_policy)
        initial_critic_params = critic_network.init(key_critic)
        initial_target_policy_params = initial_policy_params
        initial_target_critic_params = initial_critic_params

        # Create optimizers if they aren't given.
        critic_optimizer = critic_optimizer or optax.adam(1e-4)
        policy_optimizer = policy_optimizer or optax.adam(1e-4)

        # Initialize optimizers.
        initial_policy_opt_state = policy_optimizer.init(
            initial_policy_params
        )  # pytype: disable=attribute-error
        initial_critic_opt_state = critic_optimizer.init(
            initial_critic_params
        )  # pytype: disable=attribute-error

        # Create the initial state and replicate it in all devices.
        self._state = utils.replicate_in_all_devices(
            TrainingState(
                policy_params=initial_policy_params,
                target_policy_params=initial_target_policy_params,
                critic_params=initial_critic_params,
                target_critic_params=initial_target_critic_params,
                policy_opt_state=initial_policy_opt_state,
                critic_opt_state=initial_critic_opt_state,
                steps=0,
            )
        )

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None

    def step(self):
        # Sample from replay and pack the data in a Transition.
        sample = next(self._iterator)
        transitions = types.Transition(*sample.data)

        self._state, metrics = self._sgd_step(self._state, transitions)

        # Take the metrics from the first device, since they've been pmeaned over
        # all devices and are therefore identical.
        metrics = utils.get_from_first_device(metrics)

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
            "policy": self._state.target_policy_params,
            "critic": self._state.target_critic_params,
        }
        return utils.get_from_first_device([variables[name] for name in names])

    def save(self) -> TrainingState:
        return utils.get_from_first_device(self._state)

    def restore(self, state: TrainingState):
        self._state = utils.replicate_in_all_devices(state)
