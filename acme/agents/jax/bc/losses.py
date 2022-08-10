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
from typing import Callable, Optional, Tuple, Union

from acme import types
from acme.agents.jax.bc import networks as bc_networks
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.utils import loggers
import jax
import jax.numpy as jnp


loss_args = [
    bc_networks.BCNetworks, networks_lib.Params, networks_lib.PRNGKey,
    types.Transition
]
BCLossWithoutAux = Callable[loss_args, jnp.ndarray]
BCLossWithAux = Callable[loss_args, Tuple[jnp.ndarray, loggers.LoggingData]]
BCLoss = Union[BCLossWithoutAux, BCLossWithAux]


def mse() -> BCLossWithoutAux:
  """Mean Squared Error loss."""

  def loss(networks: bc_networks.BCNetworks, params: networks_lib.Params,
           key: jax_types.PRNGKey,
           transitions: types.Transition) -> jnp.ndarray:
    key, key_dropout = jax.random.split(key)
    dist_params = networks.policy_network.apply(
        params, transitions.observation, is_training=True, key=key_dropout)
    action = networks.sample_fn(dist_params, key)
    return jnp.mean(jnp.square(action - transitions.action))

  return loss


def logp() -> BCLossWithoutAux:
  """Log probability loss."""

  def loss(networks: bc_networks.BCNetworks, params: networks_lib.Params,
           key: jax_types.PRNGKey,
           transitions: types.Transition) -> jnp.ndarray:
    logits = networks.policy_network.apply(
        params, transitions.observation, is_training=True, key=key)
    logp_action = networks.log_prob(logits, transitions.action)
    return -jnp.mean(logp_action)

  return loss


def peerbc(base_loss_fn: BCLossWithoutAux, zeta: float) -> BCLossWithoutAux:
  """Peer-BC loss from https://arxiv.org/pdf/2010.01748.pdf.

  Args:
    base_loss_fn: the base loss to add RCAL on top of.
    zeta: the weight of the regularization.
  Returns:
    The loss.
  """

  def loss(networks: bc_networks.BCNetworks, params: networks_lib.Params,
           key: jax_types.PRNGKey,
           transitions: types.Transition) -> jnp.ndarray:
    key_perm, key_bc_loss, key_permuted_loss = jax.random.split(key, 3)

    permutation_keys = jax.random.split(key_perm, transitions.action.shape[0])
    permuted_actions = jax.vmap(
        jax.random.permutation, in_axes=(0, 0))(permutation_keys,
                                                transitions.action)
    permuted_transition = transitions._replace(action=permuted_actions)
    bc_loss = base_loss_fn(networks, params, key_bc_loss, transitions)
    permuted_loss = base_loss_fn(networks, params, key_permuted_loss,
                                 permuted_transition)
    return bc_loss - zeta * permuted_loss

  return loss


def rcal(base_loss_fn: BCLossWithoutAux,
         discount: float,
         alpha: float,
         num_bins: Optional[int] = None) -> BCLossWithoutAux:
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

  def loss(networks: bc_networks.BCNetworks, params: networks_lib.Params,
           key: jax_types.PRNGKey,
           transitions: types.Transition) -> jnp.ndarray:

    def logits_fn(key: jax_types.PRNGKey,
                  observations: networks_lib.Observation,
                  actions: Optional[networks_lib.Action] = None):
      logits = networks.policy_network.apply(
          params, observations, key=key, is_training=True)
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

    loss = base_loss_fn(networks, params, key, transitions)
    return loss + alpha * regularization_loss

  return loss
