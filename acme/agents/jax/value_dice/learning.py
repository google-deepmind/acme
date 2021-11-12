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

"""ValueDice learner implementation."""

import functools
import time
from typing import Any, Dict, Iterator, List, Mapping, NamedTuple, Optional, Tuple

import acme
from acme import types
from acme.agents.jax.value_dice import networks as value_dice_networks
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import optax
import reverb


class TrainingState(NamedTuple):
  """Contains training state for the learner."""
  policy_optimizer_state: optax.OptState
  policy_params: networks_lib.Params
  nu_optimizer_state: optax.OptState
  nu_params: networks_lib.Params
  key: jnp.ndarray
  steps: int


def _orthogonal_regularization_loss(params: networks_lib.Params):
  """Orthogonal regularization.

  See equation (3) in https://arxiv.org/abs/1809.11096.

  Args:
    params: Dictionary of parameters to apply regualization for.

  Returns:
    A regularization loss term.
  """
  reg_loss = 0
  for key in params:
    if isinstance(params[key], Mapping):
      reg_loss += _orthogonal_regularization_loss(params[key])
      continue
    variable = params[key]
    assert len(variable.shape) in [1, 2, 4]
    if len(variable.shape) == 1:
      # This is a bias so do not apply regularization.
      continue
    if len(variable.shape) == 4:
      # CNN
      variable = jnp.reshape(variable, (-1, variable.shape[-1]))
    prod = jnp.matmul(jnp.transpose(variable), variable)
    reg_loss += jnp.sum(jnp.square(prod * (1 - jnp.eye(prod.shape[0]))))
  return reg_loss


class ValueDiceLearner(acme.Learner):
  """ValueDice learner."""

  _state: TrainingState

  def __init__(self,
               networks: value_dice_networks.ValueDiceNetworks,
               policy_optimizer: optax.GradientTransformation,
               nu_optimizer: optax.GradientTransformation,
               discount: float,
               rng: jnp.ndarray,
               iterator_replay: Iterator[reverb.ReplaySample],
               iterator_demonstrations: Iterator[types.Transition],
               alpha: float = 0.05,
               policy_reg_scale: float = 1e-4,
               nu_reg_scale: float = 10.0,
               num_sgd_steps_per_step: int = 1,
               counter: Optional[counting.Counter] = None,
               logger: Optional[loggers.Logger] = None):

    rng, policy_key, nu_key = jax.random.split(rng, 3)
    policy_init_params = networks.policy_network.init(policy_key)
    policy_optimizer_state = policy_optimizer.init(policy_init_params)

    nu_init_params = networks.nu_network.init(nu_key)
    nu_optimizer_state = nu_optimizer.init(nu_init_params)

    def compute_losses(
        policy_params: networks_lib.Params,
        nu_params: networks_lib.Params,
        key: jnp.ndarray,
        replay_o_tm1: types.NestedArray,
        replay_a_tm1: types.NestedArray,
        replay_o_t: types.NestedArray,
        demo_o_tm1: types.NestedArray,
        demo_a_tm1: types.NestedArray,
        demo_o_t: types.NestedArray,
    ) -> jnp.ndarray:
      # TODO(damienv, hussenot): what to do with the discounts ?

      def policy(obs, key):
        dist_params = networks.policy_network.apply(policy_params, obs)
        return networks.sample(dist_params, key)

      key1, key2, key3, key4 = jax.random.split(key, 4)

      # Predicted actions.
      demo_o_t0 = demo_o_tm1
      policy_demo_a_t0 = policy(demo_o_t0, key1)
      policy_demo_a_t = policy(demo_o_t, key2)
      policy_replay_a_t = policy(replay_o_t, key3)

      replay_a_tm1 = networks.encode_action(replay_a_tm1)
      demo_a_tm1 = networks.encode_action(demo_a_tm1)
      policy_demo_a_t0 = networks.encode_action(policy_demo_a_t0)
      policy_demo_a_t = networks.encode_action(policy_demo_a_t)
      policy_replay_a_t = networks.encode_action(policy_replay_a_t)

      # "Value function" nu over the expert states.
      nu_demo_t0 = networks.nu_network.apply(nu_params, demo_o_t0,
                                             policy_demo_a_t0)
      nu_demo_tm1 = networks.nu_network.apply(nu_params, demo_o_tm1, demo_a_tm1)
      nu_demo_t = networks.nu_network.apply(nu_params, demo_o_t,
                                            policy_demo_a_t)
      nu_demo_diff = nu_demo_tm1 - discount * nu_demo_t

      # "Value function" nu over the replay buffer states.
      nu_replay_tm1 = networks.nu_network.apply(nu_params, replay_o_tm1,
                                                replay_a_tm1)
      nu_replay_t = networks.nu_network.apply(nu_params, replay_o_t,
                                              policy_replay_a_t)
      nu_replay_diff = nu_replay_tm1 - discount * nu_replay_t

      # Linear part of the loss.
      linear_loss_demo = jnp.mean(nu_demo_t0 * (1.0 - discount))
      linear_loss_rb = jnp.mean(nu_replay_diff)
      linear_loss = (linear_loss_demo * (1 - alpha) + linear_loss_rb * alpha)

      # Non linear part of the loss.
      nu_replay_demo_diff = jnp.concatenate([nu_demo_diff, nu_replay_diff],
                                            axis=0)
      replay_demo_weights = jnp.concatenate([
          jnp.ones_like(nu_demo_diff) * (1 - alpha),
          jnp.ones_like(nu_replay_diff) * alpha
      ],
                                            axis=0)
      replay_demo_weights /= jnp.mean(replay_demo_weights)
      non_linear_loss = jnp.sum(
          jax.lax.stop_gradient(
              utils.weighted_softmax(nu_replay_demo_diff, replay_demo_weights,
                                     axis=0)) *
          nu_replay_demo_diff)

      # Final loss.
      loss = (non_linear_loss - linear_loss)

      # Regularized policy loss.
      if policy_reg_scale > 0.:
        policy_reg = _orthogonal_regularization_loss(policy_params)
      else:
        policy_reg = 0.

      # Gradient penality on nu
      if nu_reg_scale > 0.0:
        batch_size = demo_o_tm1.shape[0]
        c = jax.random.uniform(key4, shape=(batch_size,))
        shape_o = [
            dim if i == 0 else 1 for i, dim in enumerate(replay_o_tm1.shape)
        ]
        shape_a = [
            dim if i == 0 else 1 for i, dim in enumerate(replay_a_tm1.shape)
        ]
        c_o = jnp.reshape(c, shape_o)
        c_a = jnp.reshape(c, shape_a)
        mixed_o_tm1 = c_o * demo_o_tm1 + (1 - c_o) * replay_o_tm1
        mixed_a_tm1 = c_a * demo_a_tm1 + (1 - c_a) * replay_a_tm1
        mixed_o_t = c_o * demo_o_t + (1 - c_o) * replay_o_t
        mixed_policy_a_t = c_a * policy_demo_a_t + (1 - c_a) * policy_replay_a_t
        mixed_o = jnp.concatenate([mixed_o_tm1, mixed_o_t], axis=0)
        mixed_a = jnp.concatenate([mixed_a_tm1, mixed_policy_a_t], axis=0)

        def sum_nu(o, a):
          return jnp.sum(networks.nu_network.apply(nu_params, o, a))

        nu_grad_o_fn = jax.grad(sum_nu, argnums=0)
        nu_grad_a_fn = jax.grad(sum_nu, argnums=1)
        nu_grad_o = nu_grad_o_fn(mixed_o, mixed_a)
        nu_grad_a = nu_grad_a_fn(mixed_o, mixed_a)
        nu_grad = jnp.concatenate([
            jnp.reshape(nu_grad_o, [batch_size, -1]),
            jnp.reshape(nu_grad_a, [batch_size, -1])], axis=-1)
        # TODO(damienv, hussenot): check for the need of eps
        # (like in the original value dice code).
        nu_grad_penalty = jnp.mean(
            jnp.square(
                jnp.linalg.norm(nu_grad + 1e-8, axis=-1, keepdims=True) - 1))
      else:
        nu_grad_penalty = 0.0

      policy_loss = -loss + policy_reg_scale * policy_reg
      nu_loss = loss + nu_reg_scale * nu_grad_penalty

      return policy_loss, nu_loss

    def sgd_step(
        state: TrainingState,
        data: Tuple[types.Transition, types.Transition]
    ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
      replay_transitions, demo_transitions = data
      key, key_loss = jax.random.split(state.key)
      compute_losses_with_input = functools.partial(
          compute_losses,
          replay_o_tm1=replay_transitions.observation,
          replay_a_tm1=replay_transitions.action,
          replay_o_t=replay_transitions.next_observation,
          demo_o_tm1=demo_transitions.observation,
          demo_a_tm1=demo_transitions.action,
          demo_o_t=demo_transitions.next_observation,
          key=key_loss)
      (policy_loss_value, nu_loss_value), vjpfun = jax.vjp(
          compute_losses_with_input,
          state.policy_params, state.nu_params)
      policy_gradients, _ = vjpfun((1.0, 0.0))
      _, nu_gradients = vjpfun((0.0, 1.0))

      # Update optimizers.
      policy_update, policy_optimizer_state = policy_optimizer.update(
          policy_gradients, state.policy_optimizer_state)
      policy_params = optax.apply_updates(state.policy_params, policy_update)

      nu_update, nu_optimizer_state = nu_optimizer.update(
          nu_gradients, state.nu_optimizer_state)
      nu_params = optax.apply_updates(state.nu_params, nu_update)

      new_state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          policy_params=policy_params,
          nu_optimizer_state=nu_optimizer_state,
          nu_params=nu_params,
          key=key,
          steps=state.steps + 1,
      )

      metrics = {
          'policy_loss': policy_loss_value,
          'nu_loss': nu_loss_value,
      }

      return new_state, metrics

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(
        'learner', asynchronous=True, serialize_fn=utils.fetch_devicearray)

    # Iterator on demonstration transitions.
    self._iterator_demonstrations = iterator_demonstrations
    self._iterator_replay = iterator_replay

    self._sgd_step = jax.jit(utils.process_multiple_batches(
        sgd_step, num_sgd_steps_per_step))

    # Create initial state.
    self._state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_init_params,
        nu_optimizer_state=nu_optimizer_state,
        nu_params=nu_init_params,
        key=rng,
        steps=0,
    )

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

  def step(self):
    # Get data from replay (dropping extras if any). Note there is no
    # extra data here because we do not insert any into Reverb.
    # TODO(raveman): Add a support for offline training, where we do not consume
    # data from the replay buffer.
    sample = next(self._iterator_replay)
    replay_transitions = types.Transition(*sample.data)

    # Get a batch of Transitions from the demonstration.
    demonstration_transitions = next(self._iterator_demonstrations)

    self._state, metrics = self._sgd_step(
        self._state, (replay_transitions, demonstration_transitions))

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Increment counts and record the current time
    counts = self._counter.increment(steps=1, walltime=elapsed_time)

    # Attempts to write the logs.
    self._logger.write({**metrics, **counts})

  def get_variables(self, names: List[str]) -> List[Any]:
    variables = {
        'policy': self._state.policy_params,
        'nu': self._state.nu_params,
    }
    return [variables[name] for name in names]

  def save(self) -> TrainingState:
    return self._state

  def restore(self, state: TrainingState):
    self._state = state
