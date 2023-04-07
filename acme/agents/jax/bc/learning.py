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

"""BC learner implementation."""

import time
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import optax

import acme
from acme import types
from acme.agents.jax.bc import losses
from acme.agents.jax.bc import networks as bc_networks
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting, loggers

_PMAP_AXIS_NAME = "data"


class TrainingState(NamedTuple):
    """Contains training state for the learner."""

    optimizer_state: optax.OptState
    policy_params: networks_lib.Params
    key: networks_lib.PRNGKey
    steps: int


def _create_loss_metrics(
    loss_has_aux: bool,
    loss_result: Union[jnp.ndarray, Tuple[jnp.ndarray, loggers.LoggingData]],
    gradients: jnp.ndarray,
):
    """Creates loss metrics for logging."""
    # Validate input.
    if loss_has_aux and not (
        len(loss_result) == 2
        and isinstance(loss_result[0], jnp.ndarray)
        and isinstance(loss_result[1], dict)
    ):
        raise ValueError(
            "Could not parse loss value and metrics from loss_fn's "
            "output. Since loss_has_aux is enabled, loss_fn must "
            "return loss_value and auxiliary metrics."
        )

    if not loss_has_aux and not isinstance(loss_result, jnp.ndarray):
        raise ValueError(
            f"Loss returns type {loss_result}. However, it should "
            "return a jnp.ndarray, given that loss_has_aux = False."
        )

    # Maybe unpack loss result.
    if loss_has_aux:
        loss, metrics = loss_result
    else:
        loss = loss_result
        metrics = {}

    # Complete metrics dict and return it.
    metrics["loss"] = loss
    metrics["gradient_norm"] = optax.global_norm(gradients)
    return metrics


class BCLearner(acme.Learner):
    """BC learner.

  This is the learning component of a BC agent. It takes a Transitions iterator
  as input and implements update functionality to learn from this iterator.
  """

    _state: TrainingState

    def __init__(
        self,
        networks: bc_networks.BCNetworks,
        random_key: networks_lib.PRNGKey,
        loss_fn: losses.BCLoss,
        optimizer: optax.GradientTransformation,
        prefetching_iterator: Iterator[types.Transition],
        num_sgd_steps_per_step: int,
        loss_has_aux: bool = False,
        logger: Optional[loggers.Logger] = None,
        counter: Optional[counting.Counter] = None,
    ):
        """Behavior Cloning Learner.

    Args:
      networks: BC networks
      random_key: RNG key.
      loss_fn: BC loss to use.
      optimizer: Optax optimizer.
      prefetching_iterator: A sharded prefetching iterator as outputted from
        `acme.jax.utils.sharded_prefetch`. Please see the documentation for
        `sharded_prefetch` for more details.
      num_sgd_steps_per_step: Number of gradient updates per step.
      loss_has_aux: Whether the loss function returns auxiliary metrics as a
        second argument.
      logger: Logger.
      counter: Counter.
    """

        def sgd_step(
            state: TrainingState, transitions: types.Transition,
        ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:

            loss_and_grad = jax.value_and_grad(loss_fn, argnums=1, has_aux=loss_has_aux)

            # Compute losses and their gradients.
            key, key_input = jax.random.split(state.key)
            loss_result, gradients = loss_and_grad(
                networks, state.policy_params, key_input, transitions
            )

            # Combine the gradient across all devices (by taking their mean).
            gradients = jax.lax.pmean(gradients, axis_name=_PMAP_AXIS_NAME)

            # Compute and combine metrics across all devices.
            metrics = _create_loss_metrics(loss_has_aux, loss_result, gradients)
            metrics = jax.lax.pmean(metrics, axis_name=_PMAP_AXIS_NAME)

            policy_update, optimizer_state = optimizer.update(
                gradients, state.optimizer_state, state.policy_params
            )
            policy_params = optax.apply_updates(state.policy_params, policy_update)

            new_state = TrainingState(
                optimizer_state=optimizer_state,
                policy_params=policy_params,
                key=key,
                steps=state.steps + 1,
            )

            return new_state, metrics

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter(prefix="learner")
        self._logger = logger or loggers.make_default_logger(
            "learner",
            asynchronous=True,
            serialize_fn=utils.fetch_devicearray,
            steps_key=self._counter.get_steps_key(),
        )

        # Split the input batch to `num_sgd_steps_per_step` minibatches in order
        # to achieve better performance on accelerators.
        sgd_step = utils.process_multiple_batches(sgd_step, num_sgd_steps_per_step)
        self._sgd_step = jax.pmap(sgd_step, axis_name=_PMAP_AXIS_NAME)

        random_key, init_key = jax.random.split(random_key)
        policy_params = networks.policy_network.init(init_key)
        optimizer_state = optimizer.init(policy_params)

        # Create initial state.
        state = TrainingState(
            optimizer_state=optimizer_state,
            policy_params=policy_params,
            key=random_key,
            steps=0,
        )
        self._state = utils.replicate_in_all_devices(state)

        self._timestamp = None

        self._prefetching_iterator = prefetching_iterator

    def step(self):
        # Get a batch of Transitions.
        transitions = next(self._prefetching_iterator)
        self._state, metrics = self._sgd_step(self._state, transitions)
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
            "policy": utils.get_from_first_device(self._state.policy_params),
        }
        return [variables[name] for name in names]

    def save(self) -> TrainingState:
        # Serialize only the first replica of parameters and optimizer state.
        return jax.tree_map(utils.get_from_first_device, self._state)

    def restore(self, state: TrainingState):
        self._state = utils.replicate_in_all_devices(state)
