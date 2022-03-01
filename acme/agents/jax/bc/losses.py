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

"""Offline losses used in variants of BC."""
from typing import Any, Callable, Optional, Tuple, Union

from acme import types
from acme.jax import networks as networks_lib
from acme.utils import loggers
import jax
import jax.numpy as jnp


ModelOutput = Any
SampleFn = Callable[[ModelOutput, networks_lib.PRNGKey], networks_lib.Action]
LogProb = jnp.ndarray
LogProbFn = Callable[[ModelOutput, networks_lib.Action], LogProb]
loss_args = [
    Callable[..., networks_lib.NetworkOutput], networks_lib.Params,
    networks_lib.PRNGKey, types.Transition
]
LossWithoutAux = Callable[loss_args, jnp.ndarray]
LossWithAux = Callable[loss_args, Tuple[jnp.ndarray, loggers.LoggingData]]
Loss = Union[LossWithoutAux, LossWithAux]


def mse(sample_fn: SampleFn) -> LossWithoutAux:
  """Mean Squared Error loss.

  Args:
    sample_fn: a method that samples an action.
  Returns:
    The loss.
  """

  def loss(apply_fn: Callable[..., networks_lib.NetworkOutput],
           params: networks_lib.Params, key: jnp.ndarray,
           transitions: types.Transition) -> jnp.ndarray:
    key, key_dropout = jax.random.split(key)
    dist_params = apply_fn(
        params, transitions.observation, is_training=True, key=key_dropout)
    action = sample_fn(dist_params, key)
    return jnp.mean(jnp.square(action - transitions.action))

  return loss


def logp(logp_fn: LogProbFn) -> LossWithoutAux:
  """Log probability loss.

  Args:
    logp_fn: a method that returns the log probability of an action.

  Returns:
    The loss.
  """

  def loss(apply_fn: Callable[..., networks_lib.NetworkOutput],
           params: networks_lib.Params, key: jnp.ndarray,
           transitions: types.Transition) -> jnp.ndarray:
    logits = apply_fn(
        params, transitions.observation, is_training=True, key=key)
    logp_action = logp_fn(logits, transitions.action)
    return -jnp.mean(logp_action)

  return loss


def peerbc(base_loss_fn: Loss, zeta: float) -> LossWithoutAux:
  """Peer-BC loss from https://arxiv.org/pdf/2010.01748.pdf.

  Args:
    base_loss_fn: the base loss to add RCAL on top of.
    zeta: the weight of the regularization.
  Returns:
    The loss.
  """

  def loss(apply_fn: Callable[..., networks_lib.NetworkOutput],
           params: networks_lib.Params, key: jnp.ndarray,
           transitions: types.Transition) -> jnp.ndarray:
    key_perm, key_bc_loss, key_permuted_loss = jax.random.split(key, 3)

    permutation_keys = jax.random.split(key_perm, transitions.action.shape[0])
    permuted_actions = jax.vmap(
        jax.random.permutation, in_axes=(0, 0))(permutation_keys,
                                                transitions.action)
    permuted_transition = transitions._replace(action=permuted_actions)
    bc_loss = base_loss_fn(apply_fn, params, key_bc_loss, transitions)
    permuted_loss = base_loss_fn(apply_fn, params, key_permuted_loss,
                                 permuted_transition)
    return bc_loss - zeta * permuted_loss

  return loss


def rcal(base_loss_fn: Loss,
         discount: float,
         alpha: float,
         num_bins: Optional[int] = None) -> LossWithoutAux:
  """https://www.cristal.univ-lille.fr/~pietquin/pdf/AAMAS_2014_BPMGOP.pdf.

  Args:
    base_loss_fn: the base loss to add RCAL on top of.
    discount: the gamma discount used in RCAL.
    alpha: the regularization parameter.
    num_bins: how many bins were used for discretization. If None the
      environment was originally discrete already.
  Returns:
    The loss function.
  """

  def loss(
      apply_fn: Callable[..., networks_lib.NetworkOutput],
      params: networks_lib.Params, key: jnp.ndarray,
      transitions: types.Transition) -> jnp.ndarray:

    def logits_fn(key, observations, actions=None):
      logits = apply_fn(params, observations, key=key, is_training=True)
      if num_bins:
        logits = jnp.reshape(logits, list(logits.shape[:-1]) + [-1, num_bins])
      if actions is None:
        actions = jnp.argmax(logits, axis=-1)
      logits_actions = jnp.sum(
          jax.nn.one_hot(actions, logits.shape[-1]) * logits, axis=-1)
      return logits_actions

    key, key1, key2 = jax.random.split(key, 3)

    logits_a_tm1 = logits_fn(key1, transitions.observation, transitions.action)
    logits_a_t = logits_fn(key2, transitions.next_observation)

    # RCAL, by making a parallel between the logits of BC and Q-values,
    # defines a regularization loss that encourages the implicit reward
    # (inferred by inversing the Bellman Equation) to be sparse.
    # NOTE: In case of discretized envs jnp.mean goes over batch and num_bins
    # dimensions.
    regularization_loss = jnp.mean(
        jnp.abs(logits_a_tm1 - discount * logits_a_t)
        )

    loss = base_loss_fn(apply_fn, params, key, transitions)
    return loss + alpha * regularization_loss

  return loss
