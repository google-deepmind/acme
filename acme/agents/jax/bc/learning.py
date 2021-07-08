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
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import acme
from acme import types
from acme.agents.jax.bc import losses
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import optax


class TrainingState(NamedTuple):
  """Contains training state for the learner."""
  optimizer_state: optax.OptState
  policy_params: networks_lib.Params
  key: networks_lib.PRNGKey
  steps: int


class BCLearner(acme.Learner):
  """BC learner.

  This is the learning component of a BC agent. It takes a Transitions iterator
  as input and implements update functionality to learn from this iterator.
  """

  _state: TrainingState

  def __init__(self,
               network: networks_lib.FeedForwardNetwork,
               random_key: networks_lib.PRNGKey,
               loss_fn: losses.Loss,
               optimizer: optax.GradientTransformation,
               demonstrations: Iterator[types.Transition],
               num_sgd_steps_per_step: int,
               logger: Optional[loggers.Logger] = None,
               counter: Optional[counting.Counter] = None):
    """Behavior Cloning Learner.

    Args:
      network: Networks with signature for apply:
        (params, obs, is_training, key) -> jnp.ndarray
        and for init:
        (rng, is_training) -> params
      random_key: RNG key.
      loss_fn: BC loss to use.
      optimizer: Optax optimizer.
      demonstrations: Demonstrations iterator.
      num_sgd_steps_per_step: Number of gradient updates per step.
      logger: Logger
      counter: Counter
    """
    def sgd_step(
        state: TrainingState,
        transitions: types.Transition,
    ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:

      loss_and_grad = jax.value_and_grad(loss_fn, argnums=1)

      # Compute losses and their gradients.
      key, key_input = jax.random.split(state.key)
      loss_value, gradients = loss_and_grad(network.apply, state.policy_params,
                                            key_input, transitions)

      policy_update, optimizer_state = optimizer.update(
          gradients, state.optimizer_state, state.policy_params)
      policy_params = optax.apply_updates(state.policy_params, policy_update)

      new_state = TrainingState(
          optimizer_state=optimizer_state,
          policy_params=policy_params,
          key=key,
          steps=state.steps + 1,
      )
      metrics = {
          'loss': loss_value,
          'gradient_norm': optax.global_norm(gradients)
      }

      return new_state, metrics

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter(prefix='learner')
    self._logger = logger or loggers.make_default_logger(
        'learner', asynchronous=True, serialize_fn=utils.fetch_devicearray)

    # Iterator on demonstration transitions.
    self._demonstrations = demonstrations

    # Split the input batch to `num_sgd_steps_per_step` minibatches in order
    # to achieve better performance on accelerators.
    self._sgd_step = jax.jit(utils.process_multiple_batches(
        sgd_step, num_sgd_steps_per_step))

    random_key, init_key = jax.random.split(random_key)
    policy_params = network.init(init_key)
    optimizer_state = optimizer.init(policy_params)

    # Create initial state.
    self._state = TrainingState(
        optimizer_state=optimizer_state,
        policy_params=policy_params,
        key=random_key,
        steps=0,
    )

    self._timestamp = None

  def step(self):
    # Get a batch of Transitions.
    transitions = next(self._demonstrations)
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
    variables = {
        'policy': self._state.policy_params,
    }
    return [variables[name] for name in names]

  def save(self) -> TrainingState:
    return self._state

  def restore(self, state: TrainingState):
    self._state = state
