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

"""Implements the MPO loss for a discrete (categorical) policy.

The MPO loss uses CategoricalMPOParams, which can be initialized using
init_params, to track the temperature and the dual variables.

Tensor shapes are annotated, where helpful, as follow:
  B: batch size,
  D: dimensionality of the action space.
"""

from typing import NamedTuple, Tuple

import distrax
import jax
import jax.numpy as jnp

_MPO_FLOAT_EPSILON = 1e-8
_MIN_LOG_TEMPERATURE = -18.0
_MIN_LOG_ALPHA = -18.0

DType = type(jnp.float32)  # _ScalarMeta, a private type.


class CategoricalMPOParams(NamedTuple):
  """NamedTuple to store trainable loss parameters."""
  log_temperature: jnp.ndarray
  log_alpha: jnp.ndarray


class CategoricalMPOStats(NamedTuple):
  """NamedTuple to store loss statistics."""
  dual_alpha: float
  dual_temperature: float

  loss_e_step: float
  loss_m_step: float
  loss_dual: float

  loss_policy: float
  loss_alpha: float
  loss_temperature: float

  kl_q_rel: float
  kl_mean_rel: float

  q_min: float
  q_max: float

  entropy_online: float
  entropy_target: float


class CategoricalMPO:
  """MPO loss for a categorical policy (Abdolmaleki et al., 2018).

  (Abdolmaleki et al., 2018): https://arxiv.org/pdf/1812.02256.pdf
  """

  def __init__(self,
               epsilon: float,
               epsilon_policy: float,
               init_log_temperature: float,
               init_log_alpha: float):
    """Initializes the MPO loss for discrete (categorical) policies.

    Args:
      epsilon: KL constraint on the non-parametric auxiliary policy, the one
        associated with the dual variable called temperature.
      epsilon_policy: KL constraint on the categorical policy, the one
        associated with the dual variable called alpha.
      init_log_temperature: initial value for the temperature in log-space, note
        a softplus (rather than an exp) will be used to transform this.
      init_log_alpha: initial value for alpha in log-space. Note that a softplus
        (rather than an exp) will be used to transform this.
    """

    # MPO constraint thresholds.
    self._epsilon = epsilon
    self._epsilon_policy = epsilon_policy

    # Initial values for the constraints' dual variables.
    self._init_log_temperature = init_log_temperature
    self._init_log_alpha = init_log_alpha

  def init_params(self, action_dim: int, dtype: DType = jnp.float32):
    """Creates an initial set of parameters."""
    del action_dim  # Unused.
    return CategoricalMPOParams(
        log_temperature=jnp.full([1], self._init_log_temperature, dtype=dtype),
        log_alpha=jnp.full([1], self._init_log_alpha, dtype=dtype))

  def __call__(
      self,
      params: CategoricalMPOParams,
      online_action_distribution: distrax.Categorical,
      target_action_distribution: distrax.Categorical,
      actions: jnp.ndarray,  # Unused.
      q_values: jnp.ndarray,  # Shape [D, B].
  ) -> Tuple[jnp.ndarray, CategoricalMPOStats]:
    """Computes the MPO loss for a categorical policy.

    Args:
      params: parameters tracking the temperature and the dual variables.
      online_action_distribution: online distribution returned by the online
        policy network; expects batch_dims of [B] and event_dims of [D].
      target_action_distribution: target distribution returned by the target
        policy network; expects same shapes as online distribution.
      actions: Unused.
      q_values: Q-values associated with every action; expects shape [D, B].

    Returns:
      Loss, combining the policy loss, KL penalty, and dual losses required to
        adapt the dual variables.
      Stats, for diagnostics and tracking performance.
    """

    q_values = jnp.transpose(q_values)  # [D, B] --> [B, D].

    # Transform dual variables from log-space.
    # Note: using softplus instead of exponential for numerical stability.
    temperature = get_temperature_from_params(params)
    alpha = jax.nn.softplus(params.log_alpha) + _MPO_FLOAT_EPSILON

    # Compute the E-step logits and the temperature loss, used to adapt the
    # tempering of Q-values.
    logits_e_step, loss_temperature = compute_weights_and_temperature_loss(  # pytype: disable=wrong-arg-types  # jax-ndarray
        q_values=q_values, logits=target_action_distribution.logits,
        epsilon=self._epsilon, temperature=temperature)
    action_distribution_e_step = distrax.Categorical(logits=logits_e_step)

    # Only needed for diagnostics: Compute estimated actualized KL between the
    # non-parametric and current target policies.
    kl_nonparametric = action_distribution_e_step.kl_divergence(
        target_action_distribution)

    # Compute the policy loss.
    loss_policy = action_distribution_e_step.cross_entropy(
        online_action_distribution)
    loss_policy = jnp.mean(loss_policy)

    # Compute the regularization.
    kl = target_action_distribution.kl_divergence(online_action_distribution)
    mean_kl = jnp.mean(kl, axis=0)
    loss_kl = jax.lax.stop_gradient(alpha) * mean_kl

    # Compute the dual loss.
    loss_alpha = alpha * (self._epsilon_policy - jax.lax.stop_gradient(mean_kl))

    # Combine losses.
    loss_dual = loss_alpha + loss_temperature
    loss = loss_policy + loss_kl + loss_dual

    # Create statistics.
    stats = CategoricalMPOStats(  # pytype: disable=wrong-arg-types  # jnp-type
        # Dual Variables.
        dual_alpha=jnp.mean(alpha),
        dual_temperature=jnp.mean(temperature),
        # Losses.
        loss_e_step=loss_policy,
        loss_m_step=loss_kl,
        loss_dual=loss_dual,
        loss_policy=jnp.mean(loss),
        loss_alpha=jnp.mean(loss_alpha),
        loss_temperature=jnp.mean(loss_temperature),
        # KL measurements.
        kl_q_rel=jnp.mean(kl_nonparametric) / self._epsilon,
        kl_mean_rel=mean_kl / self._epsilon_policy,
        # Q measurements.
        q_min=jnp.mean(jnp.min(q_values, axis=0)),
        q_max=jnp.mean(jnp.max(q_values, axis=0)),
        entropy_online=jnp.mean(online_action_distribution.entropy()),
        entropy_target=jnp.mean(target_action_distribution.entropy()),
    )

    return loss, stats


def compute_weights_and_temperature_loss(
    q_values: jnp.ndarray,
    logits: jnp.ndarray,
    epsilon: float,
    temperature: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Computes normalized importance weights for the policy optimization.

  Args:
    q_values: Q-values associated with the actions sampled from the target
      policy; expected shape [B, D].
    logits: Parameters to the categorical distribution with respect to which the
      expectations are going to be computed.
    epsilon: Desired constraint on the KL between the target and non-parametric
      policies.
    temperature: Scalar used to temper the Q-values before computing normalized
      importance weights from them. This is really the Lagrange dual variable in
      the constrained optimization problem, the solution of which is the
      non-parametric policy targeted by the policy loss.

  Returns:
    Normalized importance weights, used for policy optimization.
    Temperature loss, used to adapt the temperature.
  """

  # Temper the given Q-values using the current temperature.
  tempered_q_values = jax.lax.stop_gradient(q_values) / temperature

  # Compute the E-step normalized logits.
  unnormalized_logits = tempered_q_values + jax.nn.log_softmax(logits, axis=-1)
  logits_e_step = jax.nn.log_softmax(unnormalized_logits, axis=-1)

  # Compute the temperature loss (dual of the E-step optimization problem).
  # Note that the log normalizer will be the same for all actions, so we choose
  # only the first one.
  log_normalizer = unnormalized_logits[:, 0] - logits_e_step[:, 0]
  loss_temperature = temperature * (epsilon + jnp.mean(log_normalizer))

  return logits_e_step, loss_temperature


def clip_categorical_mpo_params(
    params: CategoricalMPOParams) -> CategoricalMPOParams:
  return params._replace(
      log_temperature=jnp.maximum(_MIN_LOG_TEMPERATURE, params.log_temperature),
      log_alpha=jnp.maximum(_MIN_LOG_ALPHA, params.log_alpha))


def get_temperature_from_params(params: CategoricalMPOParams) -> float:
  return jax.nn.softplus(params.log_temperature) + _MPO_FLOAT_EPSILON  # pytype: disable=bad-return-type  # jax-nn-types
