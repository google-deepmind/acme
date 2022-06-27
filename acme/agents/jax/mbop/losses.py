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

"""Loss function wrappers, assuming a leading batch axis."""

import dataclasses
from typing import Any, Callable, Optional, Tuple, Union

from acme import types
from acme.agents.jax.mbop import dataset
from acme.jax import networks
import jax
import jax.numpy as jnp

# The apply function takes an observation (and an action) as arguments, and is
# usually a network with bound parameters.
TransitionApplyFn = Callable[[networks.Observation, networks.Action], Any]
ObservationOnlyTransitionApplyFn = Callable[[networks.Observation], Any]
TransitionLoss = Callable[[
    Union[TransitionApplyFn, ObservationOnlyTransitionApplyFn], types.Transition
], jnp.ndarray]


def mse(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
  """MSE distance."""
  return jnp.mean(jnp.square(a - b))


def world_model_loss(apply_fn: Callable[[networks.Observation, networks.Action],
                                        Tuple[networks.Observation,
                                              jnp.ndarray]],
                     steps: types.Transition) -> jnp.ndarray:
  """Returns the loss for the world model.

  Args:
    apply_fn: applies a transition model (o_t, a_t) -> (o_t+1, r), expects the
      leading axis to index the batch and the second axis to index the
      transition triplet (t-1, t, t+1).
    steps: RLDS dictionary of transition triplets as prepared by
      `rlds_loader.episode_to_timestep_batch`.

  Returns:
    A scalar loss value as jnp.ndarray.
  """
  observation_t = jax.tree_map(lambda obs: obs[:, dataset.CURRENT, ...],
                               steps.observation)
  action_t = steps.action[:, dataset.CURRENT, ...]
  observation_tp1 = jax.tree_map(lambda obs: obs[:, dataset.NEXT, ...],
                                 steps.observation)
  reward_t = steps.reward[:, dataset.CURRENT, ...]
  (predicted_observation_tp1,
   predicted_reward_t) = apply_fn(observation_t, action_t)
  # predicted_* variables may have an extra outer dimension due to ensembling,
  # the mse loss still works due to broadcasting however.
  if len(observation_tp1.shape) != len(reward_t.shape):
    # The rewards from the transitions may not have the last singular dimension.
    reward_t = jnp.expand_dims(reward_t, axis=-1)
  return mse(
      jnp.concatenate([predicted_observation_tp1, predicted_reward_t], axis=-1),
      jnp.concatenate([observation_tp1, reward_t], axis=-1))


def policy_prior_loss(
    apply_fn: Callable[[networks.Observation, networks.Action],
                       networks.Action], steps: types.Transition):
  """Returns the loss for the policy prior.

  Args:
    apply_fn: applies a policy prior (o_t, a_t) -> a_t+1, expects the leading
      axis to index the batch and the second axis to index the transition
      triplet (t-1, t, t+1).
    steps: RLDS dictionary of transition triplets as prepared by
      `rlds_loader.episode_to_timestep_batch`.

  Returns:
    A scalar loss value as jnp.ndarray.
  """
  observation_t = jax.tree_map(lambda obs: obs[:, dataset.CURRENT, ...],
                               steps.observation)
  action_tm1 = steps.action[:, dataset.PREVIOUS, ...]
  action_t = steps.action[:, dataset.CURRENT, ...]

  predicted_action_t = apply_fn(observation_t, action_tm1)
  return mse(predicted_action_t, action_t)


def return_loss(apply_fn: Callable[[networks.Observation, networks.Action],
                                   jnp.ndarray], steps: types.Transition):
  """Returns the loss for the n-step return model.

  Args:
    apply_fn: applies an n-step return model (o_t, a_t) -> r, expects the
      leading axis to index the batch and the second axis to index the
      transition triplet (t-1, t, t+1).
    steps: RLDS dictionary of transition triplets as prepared by
      `rlds_loader.episode_to_timestep_batch`.

  Returns:
    A scalar loss value as jnp.ndarray.
  """
  observation_t = jax.tree_map(lambda obs: obs[:, dataset.CURRENT, ...],
                               steps.observation)
  action_t = steps.action[:, dataset.CURRENT, ...]
  n_step_return_t = steps.extras[dataset.N_STEP_RETURN][:, dataset.CURRENT, ...]

  predicted_n_step_return_t = apply_fn(observation_t, action_t)
  return mse(predicted_n_step_return_t, n_step_return_t)


@dataclasses.dataclass
class MBOPLosses:
  """Losses for the world model, policy prior and the n-step return."""
  world_model_loss: Optional[TransitionLoss] = world_model_loss
  policy_prior_loss: Optional[TransitionLoss] = policy_prior_loss
  n_step_return_loss: Optional[TransitionLoss] = return_loss
