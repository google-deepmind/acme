# Lint as: python3
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

"""BC Learner implementation."""

from typing import Callable, Dict, List, NamedTuple, Tuple

import acme
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from dm_env import specs
import jax
import jax.numpy as jnp
import optax
import reverb
import rlax
import tensorflow as tf

LossFn = Callable[[jnp.DeviceArray, jnp.DeviceArray], jnp.DeviceArray]


def _sparse_categorical_cross_entropy(
    labels: jnp.DeviceArray, logits: jnp.DeviceArray) -> jnp.DeviceArray:
  """Converts labels to one-hot and computes categorical cross-entropy."""
  one_hot_labels = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
  cce = jax.vmap(rlax.categorical_cross_entropy)
  return cce(one_hot_labels, logits)


class TrainingState(NamedTuple):
  """Holds the agent's training state."""
  params: networks_lib.Params
  opt_state: optax.OptState
  steps: int


class BCLearner(acme.Learner):
  """BC learner.

  This is the learning component of a BC agent. IE it takes a dataset as input
  and implements update functionality to learn from this dataset.
  """

  def __init__(self,
               network: networks_lib.FeedForwardNetwork,
               obs_spec: specs.Array,
               optimizer: optax.GradientTransformation,
               random_key: networks_lib.PRNGKey,
               dataset: tf.data.Dataset,
               loss_fn: LossFn = _sparse_categorical_cross_entropy,
               counter: counting.Counter = None,
               logger: loggers.Logger = None):
    """Initializes the learner."""

    def loss(params: networks_lib.Params,
             sample: reverb.ReplaySample) -> jnp.ndarray:
      # Pull out the data needed for updates.
      o_tm1, a_tm1, r_t, d_t, o_t = sample.data
      del r_t, d_t, o_t
      logits = network.apply(params, o_tm1)
      return jnp.mean(loss_fn(a_tm1, logits))

    def sgd_step(
        state: TrainingState, sample: reverb.ReplaySample
    ) -> Tuple[TrainingState, Dict[str, jnp.DeviceArray]]:
      """Do a step of SGD."""
      grad_fn = jax.value_and_grad(loss)
      loss_value, gradients = grad_fn(state.params, sample)
      updates, new_opt_state = optimizer.update(gradients, state.opt_state)
      new_params = optax.apply_updates(state.params, updates)

      steps = state.steps + 1

      new_state = TrainingState(
          params=new_params, opt_state=new_opt_state, steps=steps)

      # Compute the global norm of the gradients for logging.
      global_gradient_norm = optax.global_norm(gradients)
      fetches = {'loss': loss_value, 'gradient_norm': global_gradient_norm}

      return new_state, fetches

    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

    # Get an iterator over the dataset.
    self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types
    # TODO(b/155086959): Fix type stubs and remove.

    # Initialise parameters and optimiser state.
    initial_params = network.init(
        random_key, utils.add_batch_dim(utils.zeros_like(obs_spec)))
    initial_opt_state = optimizer.init(initial_params)

    self._state = TrainingState(
        params=initial_params, opt_state=initial_opt_state, steps=0)

    self._sgd_step = jax.jit(sgd_step)

  def step(self):
    batch = next(self._iterator)
    # Do a batch of SGD.
    self._state, result = self._sgd_step(self._state, batch)

    # Update our counts and record it.
    counts = self._counter.increment(steps=1)
    result.update(counts)

    self._logger.write(result)

  def get_variables(self, names: List[str]) -> List[networks_lib.Params]:
    return [self._state.params]

  def save(self) -> TrainingState:
    return self._state

  def restore(self, state: TrainingState):
    self._state = state
