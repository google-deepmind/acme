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

"""DQN learner implementation."""

from typing import Iterator, List, NamedTuple, Tuple

import acme
from acme.adders import reverb as adders
from acme.jax import networks
from acme.jax import utils
from acme.utils import async_utils
from acme.utils import counting
from acme.utils import loggers
from dm_env import specs
import haiku as hk
import jax
from jax.experimental import optix
import jax.numpy as jnp
import reverb
import rlax


class TrainingState(NamedTuple):
  """Holds the agent's training state."""
  params: hk.Params
  target_params: hk.Params
  opt_state: optix.OptState
  step: int


class LearnerOutputs(NamedTuple):
  """Outputs from the SGD step, for logging and updating priorities."""
  keys: jnp.ndarray
  priorities: jnp.ndarray


class DQNLearner(acme.Learner, acme.Saveable):
  """DQN learner."""

  _state: TrainingState

  def __init__(self,
               network: networks.QNetwork,
               obs_spec: specs.Array,
               discount: float,
               importance_sampling_exponent: float,
               target_update_period: int,
               iterator: Iterator[reverb.ReplaySample],
               optimizer: optix.InitUpdate,
               rng: hk.PRNGSequence,
               max_abs_reward: float = 1.,
               huber_loss_parameter: float = 1.,
               replay_client: reverb.Client = None,
               counter: counting.Counter = None,
               logger: loggers.Logger = None):
    """Initializes the learner."""

    # Transform network into a pure function.
    network = hk.transform(network)

    def loss(params: hk.Params, target_params: hk.Params,
             sample: reverb.ReplaySample):
      o_tm1, a_tm1, r_t, d_t, o_t = sample.data
      keys, probs = sample.info[:2]

      # Forward pass.
      q_tm1 = network.apply(params, o_tm1)
      q_t_value = network.apply(target_params, o_t)
      q_t_selector = network.apply(params, o_t)

      # Cast and clip rewards.
      d_t = (d_t * discount).astype(jnp.float32)
      r_t = jnp.clip(r_t, -max_abs_reward, max_abs_reward).astype(jnp.float32)

      # Compute double Q-learning n-step TD-error.
      batch_error = jax.vmap(rlax.double_q_learning)
      td_error = batch_error(q_tm1, a_tm1, r_t, d_t, q_t_value, q_t_selector)
      batch_loss = rlax.huber_loss(td_error, huber_loss_parameter)

      # Importance weighting.
      importance_weights = (1. / probs).astype(jnp.float32)
      importance_weights **= importance_sampling_exponent
      importance_weights /= jnp.max(importance_weights)

      # Reweight.
      mean_loss = jnp.mean(importance_weights * batch_loss)  # []

      priorities = jnp.abs(td_error).astype(jnp.float64)

      return mean_loss, (keys, priorities)

    def sgd_step(
        state: TrainingState,
        samples: reverb.ReplaySample) -> Tuple[TrainingState, LearnerOutputs]:
      grad_fn = jax.grad(loss, has_aux=True)
      gradients, (keys, priorities) = grad_fn(state.params, state.target_params,
                                              samples)
      updates, new_opt_state = optimizer.update(gradients, state.opt_state)
      new_params = optix.apply_updates(state.params, updates)

      new_state = TrainingState(
          params=new_params,
          target_params=state.target_params,
          opt_state=new_opt_state,
          step=state.step + 1)

      outputs = LearnerOutputs(keys=keys, priorities=priorities)

      return new_state, outputs

    # Internalise agent components (replay buffer, networks, optimizer).
    self._replay_client = replay_client
    self._iterator = utils.prefetch(iterator)

    # Internalise the hyperparameters.
    self._target_update_period = target_update_period

    # Internalise logging/counting objects.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

    # Initialise parameters and optimiser state.
    initial_params = network.init(
        next(rng), utils.add_batch_dim(utils.zeros_like(obs_spec)))
    initial_target_params = network.init(
        next(rng), utils.add_batch_dim(utils.zeros_like(obs_spec)))
    initial_opt_state = optimizer.init(initial_params)

    self._state = TrainingState(
        params=initial_params,
        target_params=initial_target_params,
        opt_state=initial_opt_state,
        step=0)

    self._forward = jax.jit(network.apply)
    self._sgd_step = jax.jit(sgd_step)

  def step(self):
    samples = next(self._iterator)
    # Do a batch of SGD.
    # TODO(jaslanides): Log metrics.
    self._state, outputs = self._sgd_step(self._state, samples)

    # Update our counts and record it.
    result = self._counter.increment(steps=1)

    # Periodically update target network parameters.
    if self._state.step % self._target_update_period == 0:
      self._state = self._state._replace(target_params=self._state.params)

    # Update priorities in replay.
    if self._replay_client:
      self._update_priorities(outputs)

    # Write to logs.
    self._logger.write(result)

  @async_utils.make_async
  def _update_priorities(self, outputs: LearnerOutputs):
    for key, priority in zip(outputs.keys, outputs.priorities):
      self._replay_client.mutate_priorities(
          table=adders.DEFAULT_PRIORITY_TABLE, updates={key: priority})

  def get_variables(self, names: List[str]) -> List[hk.Params]:
    return [self._state.params]

  def save(self) -> TrainingState:
    return self._state

  def restore(self, state: TrainingState):
    self._state = state
