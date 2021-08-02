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

from typing import NamedTuple

from acme import types
from acme.agents.jax.bc import losses
from acme.jax import learner_core as learner_core_lib
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax.types import PRNGKey, TrainingStepOutput, Variables  # pylint: disable=g-multiple-import
import jax
import optax


class TrainingState(NamedTuple):
  """Contains training state for the learner."""
  optimizer_state: optax.OptState
  policy_params: networks_lib.Params
  key: PRNGKey
  steps: int


def _init(network: networks_lib.FeedForwardNetwork,
          random_key: PRNGKey,
          optimizer: optax.GradientTransformation) -> TrainingState:
  """Behavior Cloning initialization of the first state."""

  random_key, init_key = jax.random.split(random_key)
  policy_params = network.init(init_key)
  optimizer_state = optimizer.init(policy_params)

  return TrainingState(
      optimizer_state=optimizer_state,
      policy_params=policy_params,
      key=random_key,
      steps=0,
  )


def _sgd_step(
    state: TrainingState,
    transitions: types.Transition,
    network: networks_lib.FeedForwardNetwork,
    optimizer: optax.GradientTransformation,
    loss_fn: losses.Loss
) -> TrainingStepOutput[TrainingState]:
  """Performs a minibatch SGD step, returning new state and metrics."""

  loss_and_grad = jax.value_and_grad(loss_fn, argnums=1)

  # Compute losses and their gradients.
  key, key_input = jax.random.split(state.key)
  loss_value, gradients = loss_and_grad(network.apply, state.policy_params,
                                        key_input, transitions)

  policy_update, optimizer_state = optimizer.update(gradients,
                                                    state.optimizer_state,
                                                    state.policy_params)
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

  return TrainingStepOutput(new_state, metrics)


def _get_variables(state: TrainingState) -> Variables:
  return {'policy': state.policy_params}


def make_bc_learner_core(
    network: networks_lib.FeedForwardNetwork,
    loss_fn: losses.Loss,
    optimizer: optax.GradientTransformation,
    num_sgd_steps_per_step: int = 1
) -> learner_core_lib.LearnerCore[types.Transition, TrainingState]:
  """Returns the learner core for BC.

  Args:
    network: Networks with signature for apply:
      (params, obs, is_training, key) -> jnp.ndarray
      and for init:
      (rng, is_training) -> params
    loss_fn: BC loss to use.
    optimizer: Optax optimizer.
    num_sgd_steps_per_step: How many gradient updates to perform per step.
  """

  def init(random_key: PRNGKey) -> TrainingState:
    return _init(random_key=random_key, network=network, optimizer=optimizer)

  def step(
      state: TrainingState,
      transitions: types.Transition,
  ) -> TrainingStepOutput[TrainingState]:
    return _sgd_step(
        state=state,
        transitions=transitions,
        network=network,
        optimizer=optimizer,
        loss_fn=loss_fn)

  step = utils.process_many_batches(step, num_batches=num_sgd_steps_per_step)

  return learner_core_lib.LearnerCore(
      init=init, step=step, get_variables=_get_variables)
