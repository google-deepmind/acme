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
import rlax
import typing_extensions
from copy import deepcopy


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
               counter: Optional[counting.Counter] = None,
               logger: Optional[loggers.Logger] = None,
               num_sgd_steps_per_step: int = 1):
    """Initialize the SGD learner."""
    self.network = network

    # Internalize the loss_fn with network.
    self._loss = jax.jit(functools.partial(loss_fn, self.network))

    # SGD performs the loss, optimizer update and periodic target net update.
    # this is a custom (non-acme) function built to support parallization
    @functools.partial(jax.pmap, axis_name='num_devices')
    def sgd_step(params, target_params, batch, opt_state):
      # todo: make this use the state rng_key?
      next_rng_key, rng_key = jax.random.split(jax.random.PRNGKey(1701))
      # Implements one SGD step of the loss and updates training state

      (loss, extra), grads = jax.value_and_grad(self._loss, has_aux=True)(
          params, params, batch, rng_key)

      grads = jax.lax.pmean(grads, axis_name='num_devices')
      loss = jax.lax.pmean(loss, axis_name='num_devices') # unnecessary for update, useful for logging

      extra.metrics.update({'total_loss': loss})

      updates, new_opt_state = optimizer.update(grads, opt_state)
      new_params = optax.apply_updates(params, updates)
      
      return new_params, new_opt_state, extra


    # note that we've completely removed their postprocess_aux and manually implemented its
    # functionality in `step()`. also means we don't call `process_multiple_batches`. adding this
    # assertion to prevent accidental confusion later on.
    assert num_sgd_steps_per_step == 1, "calls to `process_multiple_batches` have been removed, so \
          `num_sgd_steps_per_step` being >1 has no effect"

    
    # don't use `jit` with `sgd_step` because `jit` puts all the device data on a single device
    # by default, which defeats the purpose of pmap. was warned by jax compiler previously.
    # self._sgd_step = jax.jit(sgd_step)
    self._sgd_step = sgd_step

    # Internalise agent components
    self._data_iterator = utils.prefetch(data_iterator)
    self._target_update_period = target_update_period
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

    self.n_devices = jax.local_device_count()

    # Initialize the network parameters
    key_params, key_target, key_state = jax.random.split(random_key, 3)
    initial_params = self.network.init(key_params)
    initial_target_params = self.network.init(key_target)

    self.rng_key = key_state # this will only ever be `key_state`
    # params -> [[params], [params]]

    self._state = TrainingState(
      params=jax.tree_map(lambda x: jnp.array([x] * self.n_devices), initial_params),
      target_params=jax.tree_map(lambda x: jnp.array([x] * self.n_devices), initial_target_params),
      opt_state=jax.tree_map(lambda x: jnp.array([x] * self.n_devices), optimizer.init(initial_params)),
      steps=0,
      rng_key=key_state,
    )

    # Update replay priorities
    def update_priorities(reverb_update: Optional[ReverbUpdate]) -> None:
      if reverb_update is None or replay_client is None:
        return
      else:
        replay_client.mutate_priorities(
            table=adders.DEFAULT_PRIORITY_TABLE,
            updates=dict(zip(reverb_update.keys, reverb_update.priorities)))
    self._replay_client = replay_client
    self._async_priority_updater = async_utils.AsyncExecutor(update_priorities)

  def step(self):
    """Takes one SGD step on the learner."""
    batch = next(self._data_iterator)

    # reshaping of batch for pmap compatibility
    # [batchsize, ...] -> [num_devices, batchsize per device, ...]

    def fix(x, n_devices=self.n_devices):
      if len(x.shape) == 1:
        return x.reshape((n_devices, x.shape[0] // n_devices))
      else:
        return x.reshape((n_devices, x.shape[0] // n_devices, *x.shape[1:])) 

    fixed_batch = jax.tree_map(fix, batch)

    new_params, new_opt_state, extra = self._sgd_step(self._state.params, self._state.target_params, fixed_batch, self._state.opt_state)

    steps = self._state.steps + 1

    # update params periodically
    # theoretically works, but need to run it multiple steps and see if it updates
    target_params = rlax.periodic_update(self._state.params, self._state.target_params, self._state.steps, self._target_update_period)

    # reshape back to pre-pmap dimensions (otherwise not the right shape for insertion to reverb)
    reverb_update = jax.tree_map(lambda a: jnp.reshape(a, (a.shape[0]*a.shape[1])), extra.reverb_update)
    
    # taken from old `postprocess_aux`
    reverb_update = jax.tree_map(lambda a: jnp.reshape(a, (-1, *a.shape[2:])), reverb_update)
    extra = extra._replace(metrics=jax.tree_map(jnp.mean, extra.metrics), reverb_update=reverb_update)

    if self._replay_client:
      self._async_priority_updater.put(reverb_update)

    # Update our counts and record it.
    result = self._counter.increment(steps=1)
    result.update(extra.metrics)
    self._logger.write(result)

    # update internal state representation
    self._state = TrainingState(
        params=new_params,
        target_params=target_params,
        opt_state=new_opt_state,
        steps=steps,
        rng_key=self.rng_key
    )

    # print("IT WORKED BABY")
    # import sys; sys.exit(-1)

  def get_variables(self, names: List[str]) -> List[networks_lib.Params]:
    # taken from 
    return [jax.device_get(jax.tree_map(lambda x: x[0], self._state.params))]

  def save(self) -> TrainingState:
    return self._state

  def restore(self, state: TrainingState):
    self._state = state
