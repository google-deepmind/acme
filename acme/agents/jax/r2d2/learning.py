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

"""R2D2 learner implementation."""

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
import rlax
import tree

_PMAP_AXIS_NAME = 'data'


class TrainingState(NamedTuple):
  """Holds the agent's training state."""
  params: networks_lib.Params
  target_params: networks_lib.Params
  opt_state: optax.OptState
  steps: int
  random_key: networks_lib.PRNGKey


class R2D2Learner(acme.Learner):
  """R2D2 learner."""

  def __init__(self,
               unroll: networks_lib.FeedForwardNetwork,
               initial_state: networks_lib.FeedForwardNetwork,
               batch_size: int,
               random_key: networks_lib.PRNGKey,
               burn_in_length: int,
               discount: float,
               importance_sampling_exponent: float,
               max_priority_weight: float,
               target_update_period: int,
               iterator: Iterator[reverb.ReplaySample],
               optimizer: optax.GradientTransformation,
               bootstrap_n: int = 5,
               tx_pair: rlax.TxPair = rlax.SIGNED_HYPERBOLIC_PAIR,
               clip_rewards: bool = False,
               max_abs_reward: float = 1.,
               use_core_state: bool = True,
               prefetch_size: int = 2,
               replay_client: Optional[reverb.Client] = None,
               counter: Optional[counting.Counter] = None,
               logger: Optional[loggers.Logger] = None):
    """Initializes the learner."""

    random_key, key_initial_1, key_initial_2 = jax.random.split(random_key, 3)
    initial_state_params = initial_state.init(key_initial_1, batch_size)
    initial_state = initial_state.apply(initial_state_params, key_initial_2,
                                        batch_size)

    def loss(
        params: networks_lib.Params,
        target_params: networks_lib.Params,
        key_grad: networks_lib.PRNGKey,
        sample: reverb.ReplaySample
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
      """Computes mean transformed N-step loss for a batch of sequences."""

      # Convert sample data to sequence-major format [T, B, ...].
      data = utils.batch_to_sequence(sample.data)

      # Get core state & warm it up on observations for a burn-in period.
      if use_core_state:
        # Replay core state.
        online_state = jax.tree_map(lambda x: x[0], data.extras['core_state'])
      else:
        online_state = initial_state
      target_state = online_state

      # Maybe burn the core state in.
      if burn_in_length:
        burn_obs = jax.tree_map(lambda x: x[:burn_in_length], data.observation)
        key_grad, key1, key2 = jax.random.split(key_grad, 3)
        _, online_state = unroll.apply(params, key1, burn_obs, online_state)
        _, target_state = unroll.apply(target_params, key2, burn_obs,
                                       target_state)

      # Only get data to learn on from after the end of the burn in period.
      data = jax.tree_map(lambda seq: seq[burn_in_length:], data)

      # Unroll on sequences to get online and target Q-Values.
      key1, key2 = jax.random.split(key_grad)
      online_q, _ = unroll.apply(params, key1, data.observation, online_state)
      target_q, _ = unroll.apply(target_params, key2, data.observation,
                                 target_state)

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
              rlax.transformed_n_step_q_learning,
              n=bootstrap_n,
              tx_pair=tx_pair),
          in_axes=1,
          out_axes=1)
      # TODO(b/183945808): when this bug is fixed, truncations of actions,
      # rewards, and discounts will no longer be necessary.
      batch_td_error = batch_td_error_fn(
          online_q[:-1],
          data.action[:-1],
          target_q[1:],
          selector_actions[1:],
          rewards[:-1],
          discounts[:-1])
      batch_loss = 0.5 * jnp.square(batch_td_error).sum(axis=0)

      # Importance weighting.
      probs = sample.info.probability
      importance_weights = (1. / (probs + 1e-6)).astype(online_q.dtype)
      importance_weights **= importance_sampling_exponent
      importance_weights /= jnp.max(importance_weights)
      mean_loss = jnp.mean(importance_weights * batch_loss)

      # Calculate priorities as a mixture of max and mean sequence errors.
      abs_td_error = jnp.abs(batch_td_error).astype(online_q.dtype)
      max_priority = max_priority_weight * jnp.max(abs_td_error, axis=0)
      mean_priority = (1 - max_priority_weight) * jnp.mean(abs_td_error, axis=0)
      priorities = (max_priority + mean_priority)

      return mean_loss, priorities

    def sgd_step(
        state: TrainingState,
        samples: reverb.ReplaySample
    ) -> Tuple[TrainingState, jnp.ndarray, Dict[str, jnp.ndarray]]:
      """Performs an update step, averaging over pmap replicas."""

      # Compute loss and gradients.
      grad_fn = jax.value_and_grad(loss, has_aux=True)
      key, key_grad = jax.random.split(state.random_key)
      (loss_value, priorities), gradients = grad_fn(state.params,
                                                    state.target_params,
                                                    key_grad,
                                                    samples)

      # Average gradients over pmap replicas before optimizer update.
      gradients = jax.lax.pmean(gradients, _PMAP_AXIS_NAME)

      # Apply optimizer updates.
      updates, new_opt_state = optimizer.update(gradients, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)

      # Periodically update target networks.
      steps = state.steps + 1
      target_params = rlax.periodic_update(
          new_params, state.target_params, steps, self._target_update_period)

      new_state = TrainingState(
          params=new_params,
          target_params=target_params,
          opt_state=new_opt_state,
          steps=steps,
          random_key=key)
      return new_state, priorities, {'loss': loss_value}

    def update_priorities(
        keys_and_priorities: Tuple[jnp.ndarray, jnp.ndarray]):
      keys, priorities = keys_and_priorities
      keys, priorities = tree.map_structure(
          # Fetch array and combine device and batch dimensions.
          lambda x: utils.fetch_devicearray(x).reshape((-1,) + x.shape[2:]),
          (keys, priorities))
      replay_client.mutate_priorities(  # pytype: disable=attribute-error
          table=adders.DEFAULT_PRIORITY_TABLE,
          updates=dict(zip(keys, priorities)))

    # Internalise components, hyperparameters, logger, counter, and methods.
    self._replay_client = replay_client
    self._target_update_period = target_update_period
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        'learner',
        asynchronous=True,
        serialize_fn=utils.fetch_devicearray,
        time_delta=1.)

    self._sgd_step = jax.pmap(sgd_step, axis_name=_PMAP_AXIS_NAME)
    self._async_priority_updater = async_utils.AsyncExecutor(update_priorities)

    # Initialise and internalise training state (parameters/optimiser state).
    random_key, key_init = jax.random.split(random_key)
    initial_params = unroll.init(key_init, initial_state)
    opt_state = optimizer.init(initial_params)

    state = TrainingState(
        params=initial_params,
        target_params=initial_params,
        opt_state=opt_state,
        steps=jnp.array(0),
        random_key=random_key)
    # Replicate parameters.
    self._state = utils.replicate_in_all_devices(state)

    # Shard multiple inputs with on-device prefetching.
    # We split samples in two outputs, the keys which need to be kept on-host
    # since int64 arrays are not supported in TPUs, and the entire sample
    # separately so it can be sent to the sgd_step method.
    def split_sample(sample: reverb.ReplaySample) -> utils.PrefetchingSplit:
      return utils.PrefetchingSplit(host=sample.info.key, device=sample)

    self._prefetched_iterator = utils.sharded_prefetch(
        iterator,
        buffer_size=prefetch_size,
        num_threads=jax.local_device_count(),
        split_fn=split_sample)

  def step(self):
    prefetching_split = next(self._prefetched_iterator)
    # The split_sample method passed to utils.sharded_prefetch specifies what
    # parts of the objects returned by the original iterator are kept in the
    # host and what parts are prefetched on-device.
    # In this case the host property of the prefetching split contains only the
    # replay keys and the device property is the prefetched full original
    # sample.
    keys, samples = prefetching_split.host, prefetching_split.device

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
    # Return first replica of parameters.
    return [utils.get_from_first_device(self._state.params)]

  def save(self) -> TrainingState:
    # Serialize only the first replica of parameters and optimizer state.
    return jax.tree_map(utils.get_from_first_device, self._state)

  def restore(self, state: TrainingState):
    self._state = utils.replicate_in_all_devices(state)
