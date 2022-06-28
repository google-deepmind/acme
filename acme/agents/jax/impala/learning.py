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
      entropy_cost: float = 0.,
      baseline_cost: float = 1.,
      max_abs_reward: float = np.inf,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      devices: Optional[Sequence[jax.xla.Device]] = None,
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

    loss_fn = losses.impala_loss(
        networks.unroll_fn,
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
      key, key_initial_state = jax.random.split(key)
      # Note: parameters do not depend on the batch size, so initial_state below
      # does not need a batch dimension.
      # TODO(jferret): as it stands, we do not yet support
      # training the initial state params.
      initial_state = networks.initial_state_fn(key_initial_state)

      initial_params = networks.unroll_init_fn(key, initial_state)
      initial_opt_state = optimizer.init(initial_params)
      return TrainingState(
          params=initial_params, opt_state=initial_opt_state)

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
    return [utils.get_from_first_device(self._state.params, as_numpy=False)]

  def save(self) -> TrainingState:
    # Serialize only the first replica of parameters and optimizer state.
    return jax.tree_map(utils.get_from_first_device, self._state)

  def restore(self, state: TrainingState):
    self._state = utils.replicate_in_all_devices(state, self._local_devices)
