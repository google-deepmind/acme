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

import time
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

from absl import logging
import acme
from acme.agents.jax.impala import networks as impala_networks
from acme.jax import losses
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb

_PMAP_AXIS_NAME = 'data'


class TrainingState(NamedTuple):
  """Training state consists of network parameters and optimiser state."""
  params: networks_lib.Params
  opt_state: optax.OptState


class IMPALALearner(acme.Learner):
  """Learner for an importanced-weighted advantage actor-critic."""

  def __init__(
      self,
      networks: impala_networks.IMPALANetworks,
      iterator: Iterator[reverb.ReplaySample],
      optimizer: optax.GradientTransformation,
      random_key: networks_lib.PRNGKey,
      discount: float = 0.99,
      entropy_cost: float = 0.0,
      baseline_cost: float = 1.0,
      max_abs_reward: float = np.inf,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      devices: Optional[Sequence[jax.Device]] = None,
      prefetch_size: int = 2,
  ):
    local_devices = jax.local_devices()
    process_id = jax.process_index()
    logging.info('Learner process id: %s. Devices passed: %s', process_id,
                 devices)
    logging.info('Learner process id: %s. Local devices from JAX API: %s',
                 process_id, local_devices)
    self._devices = devices or local_devices
    self._local_devices = [d for d in self._devices if d in local_devices]

    self._iterator = iterator

    def unroll_without_rng(
        params: networks_lib.Params, observations: networks_lib.Observation,
        initial_state: networks_lib.RecurrentState
    ) -> Tuple[networks_lib.NetworkOutput, networks_lib.RecurrentState]:
      unused_rng = jax.random.PRNGKey(0)
      return networks.unroll(params, unused_rng, observations, initial_state)

    loss_fn = losses.impala_loss(
        # TODO(b/244319884): Consider supporting the use of RNG in impala_loss.
        unroll_fn=unroll_without_rng,
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
      grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
      (loss_value, metrics), gradients = grad_fn(state.params, sample)

      # Average gradients over pmap replicas before optimizer update.
      gradients = jax.lax.pmean(gradients, _PMAP_AXIS_NAME)

      # Apply updates.
      updates, new_opt_state = optimizer.update(gradients, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)

      metrics.update({
          'loss': loss_value,
          'param_norm': optax.global_norm(new_params),
          'param_updates_norm': optax.global_norm(updates),
      })

      new_state = TrainingState(params=new_params, opt_state=new_opt_state)

      return new_state, metrics

    def make_initial_state(key: jnp.ndarray) -> TrainingState:
      """Initialises the training state (parameters and optimiser state)."""
      initial_params = networks.init(key)
      return TrainingState(
          params=initial_params, opt_state=optimizer.init(initial_params))

    # Initialise training state (parameters and optimiser state).
    state = make_initial_state(random_key)
    self._state = utils.replicate_in_all_devices(state, self._local_devices)

    self._sgd_step = jax.pmap(
        sgd_step, axis_name=_PMAP_AXIS_NAME, devices=self._devices)

    # Set up logging/counting.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        'learner', steps_key=self._counter.get_steps_key())

  def step(self):
    """Does a step of SGD and logs the results."""
    samples = next(self._iterator)

    # Do a batch of SGD.
    start = time.time()
    self._state, results = self._sgd_step(self._state, samples)

    # Take results from first replica.
    # NOTE: This measure will be a noisy estimate for the purposes of the logs
    # as it does not pmean over all devices.
    results = utils.get_from_first_device(results)

    # Update our counts and record them.
    counts = self._counter.increment(steps=1, time_elapsed=time.time() - start)

    # Maybe write logs.
    self._logger.write({**results, **counts})

  def get_variables(self, names: Sequence[str]) -> List[networks_lib.Params]:
    # Return first replica of parameters.
    return utils.get_from_first_device([self._state.params], as_numpy=False)

  def save(self) -> TrainingState:
    # Serialize only the first replica of parameters and optimizer state.
    return utils.get_from_first_device(self._state)

  def restore(self, state: TrainingState):
    self._state = utils.replicate_in_all_devices(state, self._local_devices)
