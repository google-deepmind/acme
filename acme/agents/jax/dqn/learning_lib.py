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

"""SgdLearner takes steps of SGD on a LossFn."""

import functools
import time
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import acme
from acme.adders import reverb as adders
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import async_utils
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import optax
import reverb
import tree
import typing_extensions


class ReverbUpdate(NamedTuple):
  """Tuple for updating reverb priority information."""
  keys: jnp.ndarray
  priorities: jnp.ndarray


class LossExtra(NamedTuple):
  """Extra information that is returned along with loss value."""
  metrics: Dict[str, jnp.DeviceArray]
  reverb_update: Optional[ReverbUpdate] = None


class LossFn(typing_extensions.Protocol):
  """A LossFn calculates a loss on a single batch of data."""

  def __call__(self,
               network: networks_lib.FeedForwardNetwork,
               params: networks_lib.Params,
               target_params: networks_lib.Params,
               batch: reverb.ReplaySample,
               key: networks_lib.PRNGKey) -> Tuple[jnp.DeviceArray, LossExtra]:
    """Calculates a loss on a single batch of data."""


class TrainingState(NamedTuple):
  """Holds the agent's training state."""
  params: networks_lib.Params
  target_params: networks_lib.Params
  opt_state: optax.OptState
  steps: int
  rng_key: networks_lib.PRNGKey


class SGDLearner(acme.Learner):
  """An Acme learner based around SGD on batches.

  This learner currently supports optional prioritized replay and assumes a
  TrainingState as described above.
  """

  def __init__(self,
               network: networks_lib.FeedForwardNetwork,
               loss_fn: LossFn,
               optimizer: optax.GradientTransformation,
               data_iterator: Iterator[reverb.ReplaySample],
               target_update_period: int,
               random_key: networks_lib.PRNGKey,
               replay_client: Optional[reverb.Client] = None,
               replay_table_name: str = adders.DEFAULT_PRIORITY_TABLE,
               counter: Optional[counting.Counter] = None,
               logger: Optional[loggers.Logger] = None,
               num_sgd_steps_per_step: int = 1):
    """Initialize the SGD learner."""
    self.network = network

    # Internalize the loss_fn with network.
    self._loss = jax.jit(functools.partial(loss_fn, self.network))

    # SGD performs the loss, optimizer update and periodic target net update.
    def sgd_step(state: TrainingState,
                 batch: reverb.ReplaySample) -> Tuple[TrainingState, LossExtra]:
      next_rng_key, rng_key = jax.random.split(state.rng_key)
      # Implements one SGD step of the loss and updates training state
      (loss, extra), grads = jax.value_and_grad(self._loss, has_aux=True)(
          state.params, state.target_params, batch, rng_key)
      extra.metrics.update({'total_loss': loss})

      # Apply the optimizer updates
      updates, new_opt_state = optimizer.update(grads, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)

      # Periodically update target networks.
      steps = state.steps + 1
      target_params = optax.periodic_update(new_params, state.target_params,
                                            steps, target_update_period)
      new_training_state = TrainingState(
          new_params, target_params, new_opt_state, steps, next_rng_key)
      return new_training_state, extra

    def postprocess_aux(extra: LossExtra) -> LossExtra:
      reverb_update = jax.tree_map(lambda a: jnp.reshape(a, (-1, *a.shape[2:])),
                                   extra.reverb_update)
      return extra._replace(
          metrics=jax.tree_map(jnp.mean, extra.metrics),
          reverb_update=reverb_update)

    self._num_sgd_steps_per_step = num_sgd_steps_per_step
    sgd_step = utils.process_multiple_batches(sgd_step, num_sgd_steps_per_step,
                                              postprocess_aux)
    self._sgd_step = jax.jit(sgd_step)

    # Internalise agent components
    self._data_iterator = data_iterator
    self._target_update_period = target_update_period
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

    # Initialize the network parameters
    key_params, key_target, key_state = jax.random.split(random_key, 3)
    initial_params = self.network.init(key_params)
    initial_target_params = self.network.init(key_target)
    self._state = TrainingState(
        params=initial_params,
        target_params=initial_target_params,
        opt_state=optimizer.init(initial_params),
        steps=0,
        rng_key=key_state,
    )

    # Update replay priorities
    def update_priorities(reverb_update: ReverbUpdate) -> None:
      if replay_client is None:
        return
      keys, priorities = tree.map_structure(
          utils.fetch_devicearray,
          (reverb_update.keys, reverb_update.priorities))
      replay_client.mutate_priorities(
          table=replay_table_name,
          updates=dict(zip(keys, priorities)))
    self._replay_client = replay_client
    self._async_priority_updater = async_utils.AsyncExecutor(update_priorities)

  def step(self):
    """Takes one SGD step on the learner."""
    with jax.profiler.StepTraceAnnotation('step',
                                          step_num=self._state.steps):
      batch = next(self._data_iterator)
      self._state, extra = self._sgd_step(self._state, batch)

      # Compute elapsed time.
      timestamp = time.time()
      elapsed = timestamp - self._timestamp if self._timestamp else 0
      self._timestamp = timestamp

      if self._replay_client and extra.reverb_update:
        reverb_update = extra.reverb_update._replace(keys=batch.info.key)
        self._async_priority_updater.put(reverb_update)

      steps_per_sec = (self._num_sgd_steps_per_step / elapsed) if elapsed else 0
      extra.metrics['steps_per_second'] = steps_per_sec

      # Update our counts and record it.
      result = self._counter.increment(
          steps=self._num_sgd_steps_per_step, walltime=elapsed)
      result.update(extra.metrics)
      self._logger.write(result)

  def get_variables(self, names: List[str]) -> List[networks_lib.Params]:
    return [self._state.params]

  def save(self) -> TrainingState:
    return self._state

  def restore(self, state: TrainingState):
    self._state = state
