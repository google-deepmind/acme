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

"""Learner for the IMPALA actor-critic agent."""

from typing import Callable, Dict, Iterator, List, NamedTuple, Tuple

import acme
from acme import specs
from acme.jax import losses
from acme.jax import networks
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb


class TrainingState(NamedTuple):
  """Training state consists of network parameters and optimiser state."""
  params: hk.Params
  opt_state: optax.OptState


class IMPALALearner(acme.Learner, acme.Saveable):
  """Learner for an importanced-weighted advantage actor-critic."""

  def __init__(
      self,
      obs_spec: specs.Array,
      unroll_fn: networks.PolicyValueRNN,
      initial_state_fn: Callable[[], hk.LSTMState],
      iterator: Iterator[reverb.ReplaySample],
      optimizer: optax.GradientTransformation,
      rng: hk.PRNGSequence,
      discount: float = 0.99,
      entropy_cost: float = 0.,
      baseline_cost: float = 1.,
      max_abs_reward: float = np.inf,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
  ):

    # Transform into pure functions.
    unroll_fn = hk.without_apply_rng(hk.transform(unroll_fn, apply_rng=True))
    initial_state_fn = hk.without_apply_rng(
        hk.transform(initial_state_fn, apply_rng=True))

    loss_fn = losses.impala_loss(
        unroll_fn,
        discount=discount,
        max_abs_reward=max_abs_reward,
        baseline_cost=baseline_cost,
        entropy_cost=entropy_cost)

    @jax.jit
    def sgd_step(
        state: TrainingState, sample: reverb.ReplaySample
    ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
      """Computes an SGD step, returning new state and metrics for logging."""

      # Compute gradients.
      grad_fn = jax.value_and_grad(loss_fn)
      loss_value, gradients = grad_fn(state.params, sample)

      # Apply updates.
      updates, new_opt_state = optimizer.update(gradients, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)

      metrics = {
          'loss': loss_value,
      }

      new_state = TrainingState(params=new_params, opt_state=new_opt_state)

      return new_state, metrics

    def make_initial_state(key: jnp.ndarray) -> TrainingState:
      """Initialises the training state (parameters and optimiser state)."""
      dummy_obs = utils.zeros_like(obs_spec)
      dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.
      initial_state = initial_state_fn.apply(None)
      initial_params = unroll_fn.init(key, dummy_obs, initial_state)
      initial_opt_state = optimizer.init(initial_params)
      return TrainingState(
          params=initial_params, opt_state=initial_opt_state)

    # Initialise training state (parameters and optimiser state).
    self._state = make_initial_state(next(rng))

    # Internalise iterator.
    self._iterator = iterator
    self._sgd_step = sgd_step

    # Set up logging/counting.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger('learner')

  def step(self):
    """Does a step of SGD and logs the results."""

    # Do a batch of SGD.
    sample = next(self._iterator)
    self._state, results = self._sgd_step(self._state, sample)

    # Update our counts and record it.
    counts = self._counter.increment(steps=1)

    # Snapshot and attempt to write logs.
    self._logger.write({**results, **counts})

  def get_variables(self, names: List[str]) -> List[hk.Params]:
    return [self._state.params]

  def save(self) -> TrainingState:
    return self._state

  def restore(self, state: TrainingState):
    self._state = state
