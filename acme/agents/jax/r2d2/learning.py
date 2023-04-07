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

"""R2D2 learner implementation."""

import functools
import time
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
import reverb
import rlax
import tree
from absl import logging

import acme
from acme.adders import reverb as adders
from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import async_utils, counting, loggers

_PMAP_AXIS_NAME = "data"
# This type allows splitting a sample between the host and device, which avoids
# putting item keys (uint64) on device for the purposes of priority updating.
R2D2ReplaySample = utils.PrefetchingSplit


class TrainingState(NamedTuple):
    """Holds the agent's training state."""

    params: networks_lib.Params
    target_params: networks_lib.Params
    opt_state: optax.OptState
    steps: int
    random_key: networks_lib.PRNGKey


class R2D2Learner(acme.Learner):
    """R2D2 learner."""

    def __init__(
        self,
        networks: r2d2_networks.R2D2Networks,
        batch_size: int,
        random_key: networks_lib.PRNGKey,
        burn_in_length: int,
        discount: float,
        importance_sampling_exponent: float,
        max_priority_weight: float,
        target_update_period: int,
        iterator: Iterator[R2D2ReplaySample],
        optimizer: optax.GradientTransformation,
        bootstrap_n: int = 5,
        tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR,
        clip_rewards: bool = False,
        max_abs_reward: float = 1.0,
        use_core_state: bool = True,
        prefetch_size: int = 2,
        replay_client: Optional[reverb.Client] = None,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
    ):
        """Initializes the learner."""

        def loss(
            params: networks_lib.Params,
            target_params: networks_lib.Params,
            key_grad: networks_lib.PRNGKey,
            sample: reverb.ReplaySample,
        ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
            """Computes mean transformed N-step loss for a batch of sequences."""

            # Get core state & warm it up on observations for a burn-in period.
            if use_core_state:
                # Replay core state.
                # NOTE: We may need to recover the type of the hk.LSTMState if the user
                # specifies a dynamically unrolled RNN as it will strictly enforce the
                # match between input/output state types.
                online_state = utils.maybe_recover_lstm_type(
                    sample.data.extras.get("core_state")
                )
            else:
                key_grad, initial_state_rng = jax.random.split(key_grad)
                online_state = networks.init_recurrent_state(
                    initial_state_rng, batch_size
                )
            target_state = online_state

            # Convert sample data to sequence-major format [T, B, ...].
            data = utils.batch_to_sequence(sample.data)

            # Maybe burn the core state in.
            if burn_in_length:
                burn_obs = jax.tree_map(lambda x: x[:burn_in_length], data.observation)
                key_grad, key1, key2 = jax.random.split(key_grad, 3)
                _, online_state = networks.unroll(params, key1, burn_obs, online_state)
                _, target_state = networks.unroll(
                    target_params, key2, burn_obs, target_state
                )

            # Only get data to learn on from after the end of the burn in period.
            data = jax.tree_map(lambda seq: seq[burn_in_length:], data)

            # Unroll on sequences to get online and target Q-Values.
            key1, key2 = jax.random.split(key_grad)
            online_q, _ = networks.unroll(params, key1, data.observation, online_state)
            target_q, _ = networks.unroll(
                target_params, key2, data.observation, target_state
            )

            # Get value-selector actions from online Q-values for double Q-learning.
            selector_actions = jnp.argmax(online_q, axis=-1)
            # Preprocess discounts & rewards.
            discounts = (data.discount * discount).astype(online_q.dtype)
            rewards = data.reward
            if clip_rewards:
                rewards = jnp.clip(rewards, -max_abs_reward, max_abs_reward)
            rewards = rewards.astype(online_q.dtype)

            # Get N-step transformed TD error and loss.
            batch_td_error_fn = jax.vmap(
                functools.partial(
                    rlax.transformed_n_step_q_learning, n=bootstrap_n, tx_pair=tx_pair
                ),
                in_axes=1,
                out_axes=1,
            )
            batch_td_error = batch_td_error_fn(
                online_q[:-1],
                data.action[:-1],
                target_q[1:],
                selector_actions[1:],
                rewards[:-1],
                discounts[:-1],
            )
            batch_loss = 0.5 * jnp.square(batch_td_error).sum(axis=0)

            # Importance weighting.
            probs = sample.info.probability
            importance_weights = (1.0 / (probs + 1e-6)).astype(online_q.dtype)
            importance_weights **= importance_sampling_exponent
            importance_weights /= jnp.max(importance_weights)
            mean_loss = jnp.mean(importance_weights * batch_loss)

            # Calculate priorities as a mixture of max and mean sequence errors.
            abs_td_error = jnp.abs(batch_td_error).astype(online_q.dtype)
            max_priority = max_priority_weight * jnp.max(abs_td_error, axis=0)
            mean_priority = (1 - max_priority_weight) * jnp.mean(abs_td_error, axis=0)
            priorities = max_priority + mean_priority

            return mean_loss, priorities

        def sgd_step(
            state: TrainingState, samples: reverb.ReplaySample
        ) -> Tuple[TrainingState, jnp.ndarray, Dict[str, jnp.ndarray]]:
            """Performs an update step, averaging over pmap replicas."""

            # Compute loss and gradients.
            grad_fn = jax.value_and_grad(loss, has_aux=True)
            key, key_grad = jax.random.split(state.random_key)
            (loss_value, priorities), gradients = grad_fn(
                state.params, state.target_params, key_grad, samples
            )

            # Average gradients over pmap replicas before optimizer update.
            gradients = jax.lax.pmean(gradients, _PMAP_AXIS_NAME)

            # Apply optimizer updates.
            updates, new_opt_state = optimizer.update(gradients, state.opt_state)
            new_params = optax.apply_updates(state.params, updates)

            # Periodically update target networks.
            steps = state.steps + 1
            target_params = optax.periodic_update(
                new_params,
                state.target_params,  # pytype: disable=wrong-arg-types  # numpy-scalars
                steps,
                self._target_update_period,
            )

            new_state = TrainingState(
                params=new_params,
                target_params=target_params,
                opt_state=new_opt_state,
                steps=steps,
                random_key=key,
            )
            return new_state, priorities, {"loss": loss_value}

        def update_priorities(keys_and_priorities: Tuple[jnp.ndarray, jnp.ndarray]):
            keys, priorities = keys_and_priorities
            keys, priorities = tree.map_structure(
                # Fetch array and combine device and batch dimensions.
                lambda x: utils.fetch_devicearray(x).reshape((-1,) + x.shape[2:]),
                (keys, priorities),
            )
            replay_client.mutate_priorities(  # pytype: disable=attribute-error
                table=adders.DEFAULT_PRIORITY_TABLE, updates=dict(zip(keys, priorities))
            )

        # Internalise components, hyperparameters, logger, counter, and methods.
        self._iterator = iterator
        self._replay_client = replay_client
        self._target_update_period = target_update_period
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger(
            "learner",
            asynchronous=True,
            serialize_fn=utils.fetch_devicearray,
            time_delta=1.0,
            steps_key=self._counter.get_steps_key(),
        )

        self._sgd_step = jax.pmap(sgd_step, axis_name=_PMAP_AXIS_NAME)
        self._async_priority_updater = async_utils.AsyncExecutor(update_priorities)

        # Initialise and internalise training state (parameters/optimiser state).
        random_key, key_init = jax.random.split(random_key)
        initial_params = networks.init(key_init)
        opt_state = optimizer.init(initial_params)

        # Log how many parameters the network has.
        sizes = tree.map_structure(jnp.size, initial_params)
        logging.info("Total number of params: %d", sum(tree.flatten(sizes.values())))

        state = TrainingState(
            params=initial_params,
            target_params=initial_params,
            opt_state=opt_state,
            steps=jnp.array(0),
            random_key=random_key,
        )
        # Replicate parameters.
        self._state = utils.replicate_in_all_devices(state)

    def step(self):
        prefetching_split = next(self._iterator)
        # The split_sample method passed to utils.sharded_prefetch specifies what
        # parts of the objects returned by the original iterator are kept in the
        # host and what parts are prefetched on-device.
        # In this case the host property of the prefetching split contains only the
        # replay keys and the device property is the prefetched full original
        # sample.
        keys = prefetching_split.host
        samples: reverb.ReplaySample = prefetching_split.device

        # Do a batch of SGD.
        start = time.time()
        self._state, priorities, metrics = self._sgd_step(self._state, samples)
        # Take metrics from first replica.
        metrics = utils.get_from_first_device(metrics)
        # Update our counts and record it.
        counts = self._counter.increment(steps=1, time_elapsed=time.time() - start)

        # Update priorities in replay.
        if self._replay_client:
            self._async_priority_updater.put((keys, priorities))

        # Attempt to write logs.
        self._logger.write({**metrics, **counts})

    def get_variables(self, names: List[str]) -> List[networks_lib.Params]:
        del names  # There's only one available set of params in this agent.
        # Return first replica of parameters.
        return utils.get_from_first_device([self._state.params])

    def save(self) -> TrainingState:
        # Serialize only the first replica of parameters and optimizer state.
        return utils.get_from_first_device(self._state)

    def restore(self, state: TrainingState):
        self._state = utils.replicate_in_all_devices(state)
