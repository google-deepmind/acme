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

"""Implements the WPO loss.

The WPO loss uses WPOParams, which can be initialized using init_params,
to track the temperature and the dual variables.

Tensor shapes are annotated, where helpful, as follow:
  B: batch size,
  N: number of sampled actions, see WPO paper for more details,
  D: dimensionality of the action space.
"""

import enum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

_WPO_FLOAT_EPSILON = 1e-8
_MIN_LOG_ALPHA = -18.0

Shape = tuple[int]
DType = type(jnp.float32)  # _ScalarMeta, a private type.


class WPOSquashingType(enum.Enum):
  """Types of squashing functions that are supported."""
  IDENTITY = 'identity'
  CBRT = 'cube_root'


class WPOParams(NamedTuple):
  """NamedTuple to store trainable loss parameters."""
  log_alpha_mean: jnp.ndarray
  log_alpha_stddev: jnp.ndarray


class WPOStats(NamedTuple):
  """NamedTuple to store loss statistics."""
  dual_alpha_mean: float | jnp.ndarray
  dual_alpha_stddev: float | jnp.ndarray

  loss_policy: float | jnp.ndarray
  loss_alpha: float | jnp.ndarray

  kl_mean_rel: float | jnp.ndarray
  kl_stddev_rel: float | jnp.ndarray

  q_min: float | jnp.ndarray
  q_max: float | jnp.ndarray

  pi_stddev_min: float | jnp.ndarray
  pi_stddev_max: float | jnp.ndarray
  pi_stddev_cond: float | jnp.ndarray


class WPO:
  """WPO loss with decoupled KL constraints as in (Pfau et al., 2025).

  This implementation is largely forked from the MPO loss, with simplifications.

  (Pfau et al., 2025): https://arxiv.org/pdf/2505.00663
  """

  def __init__(self,
               epsilon_mean: float,
               epsilon_stddev: float,
               init_log_alpha_mean: float,
               init_log_alpha_stddev: float,
               policy_loss_scale: float = 1.0,
               kl_loss_scale: float = 1.0,
               dual_loss_scale: float = 1.0,
               per_dim_constraining: bool = True,
               squashing_type: WPOSquashingType = WPOSquashingType.IDENTITY):
    """Initialize and configure the MPO loss.

    Args:
      epsilon_mean: KL constraint on the mean of the Gaussian policy, the one
        associated with the dual variable called alpha_mean.
      epsilon_stddev: KL constraint on the stddev of the Gaussian policy, the
        one associated with the dual variable called alpha_mean.
      init_log_alpha_mean: initial value for the alpha_mean in log-space, note a
        softplus (rather than an exp) will be used to transform this.
      init_log_alpha_stddev: initial value for the alpha_stddev in log-space,
        note a softplus (rather than an exp) will be used to transform this.
      policy_loss_scale: weight for the policy loss in the total loss.
      kl_loss_scale: weight for the KL loss in the total loss.
      dual_loss_scale: weight for the dual loss in the total loss.
      per_dim_constraining: whether to enforce the KL constraint on each
        dimension independently; this is the default. Otherwise the overall KL
        is constrained, which allows some dimensions to change more at the
        expense of others staying put.
      squashing_type: Enum of different functions available for squashing the
        action-value gradient in the policy loss. Default is identity.
    """

    # WPO constrain thresholds.
    self._epsilon_mean = epsilon_mean
    self._epsilon_stddev = epsilon_stddev

    # Initial values for the constraints' dual variables.
    self._init_log_alpha_mean = init_log_alpha_mean
    self._init_log_alpha_stddev = init_log_alpha_stddev

    # Relative weights for the various losses.
    self._policy_loss_scale = policy_loss_scale
    self._kl_loss_scale = kl_loss_scale
    self._dual_loss_scale = dual_loss_scale

    # Whether to ensure per-dimension KL constraint satisfication.
    self._per_dim_constraining = per_dim_constraining
    self._squashing_type = squashing_type

  @property
  def per_dim_constraining(self):
    return self._per_dim_constraining

  def init_params(self, action_dim: int, dtype: DType = jnp.float32):
    """Creates an initial set of parameters."""

    if self._per_dim_constraining:
      dual_variable_shape = [action_dim]
    else:
      dual_variable_shape = [1]

    log_alpha_mean = jnp.full(
        dual_variable_shape, self._init_log_alpha_mean, dtype=dtype)

    log_alpha_stddev = jnp.full(
        dual_variable_shape, self._init_log_alpha_stddev, dtype=dtype)

    return WPOParams(
        log_alpha_mean=log_alpha_mean,
        log_alpha_stddev=log_alpha_stddev)

  def __call__(
      self,
      params: WPOParams,
      online_action_distribution: tfd.MultivariateNormalDiag |tfd.Independent,
      target_action_distribution: tfd.MultivariateNormalDiag | tfd.Independent,
      actions: jnp.ndarray,  # Shape [N, B, D].
      q_values: jnp.ndarray,  # Shape [N, B].
      q_values_grad: jnp.ndarray,  # Shape [N, B, D].
      is_terminal: jnp.ndarray,  # Shape [B, T].
  ) -> tuple[jnp.ndarray, WPOStats]:
    """Computes the decoupled WPO loss.

    Args:
      params: parameters tracking the temperature and the dual variables.
      online_action_distribution: online distribution returned by the online
        policy network; expects batch_dims of [B] and event_dims of [D].
      target_action_distribution: target distribution returned by the target
        policy network; expects same shapes as online distribution.
      actions: actions sampled from the policy; expects shape [N, B, D].
      q_values: Q-values associated with each action; expects shape [N, B].
      q_values_grad: gradient of the Q-values with respect to the actions;
        expects shape [N, B, D].
      is_terminal: boolean array indicating whether the state is terminal;
        expects shape [B].

    Returns:
      Loss, combining the policy loss, KL penalty, and dual losses required to
        adapt the dual variables.
      Stats, for diagnostics and tracking performance.
    """

    # Cast `MultivariateNormalDiag`s to Independent Normals.
    # The latter allows us to satisfy KL constraints per-dimension.
    if isinstance(target_action_distribution, tfd.MultivariateNormalDiag):
      target_action_distribution = tfd.Independent(
          tfd.Normal(target_action_distribution.mean(),
                     target_action_distribution.stddev()))
      online_action_distribution = tfd.Independent(
          tfd.Normal(online_action_distribution.mean(),
                     online_action_distribution.stddev()))

    # Transform dual variables from log-space.
    # Note: using softplus instead of exponential for numerical stability.
    alpha_mean = jax.nn.softplus(params.log_alpha_mean) + _WPO_FLOAT_EPSILON
    alpha_stddev = jax.nn.softplus(params.log_alpha_stddev) + _WPO_FLOAT_EPSILON

    # Get online and target means and stddevs in preparation for decomposition.
    online_mean = online_action_distribution.distribution.mean()
    online_scale = online_action_distribution.distribution.stddev()
    target_mean = target_action_distribution.distribution.mean()
    target_scale = target_action_distribution.distribution.stddev()

    # Decompose the online policy into fixed-mean & fixed-stddev distributions.
    # This has been documented as having better performance in bandit settings,
    # see e.g. https://arxiv.org/pdf/1812.02256.pdf.
    fixed_stddev_distribution = tfd.Independent(
        tfd.Normal(loc=online_mean, scale=target_scale))
    fixed_mean_distribution = tfd.Independent(
        tfd.Normal(loc=target_mean, scale=online_scale))

    # Compute the policy losses.
    # Wrap the action distribution with natural gradient adaptor so that
    # gradients are scaled according to the variance (but leave the forward
    # pass unchanged)
    loss_policy = compute_wpo_loss(
        actions, q_values_grad,
        natural_gradient_adaptor(online_action_distribution),
        self._squashing_type,
        is_terminal)

    # Compute the decomposed KL between the target and online policies.
    if self._per_dim_constraining:
      kl_mean = target_action_distribution.distribution.kl_divergence(
          fixed_stddev_distribution.distribution)  # Shape [B, D].
      kl_stddev = target_action_distribution.distribution.kl_divergence(
          fixed_mean_distribution.distribution)  # Shape [B, D].
    else:
      kl_mean = target_action_distribution.kl_divergence(
          fixed_stddev_distribution)  # Shape [B].
      kl_stddev = target_action_distribution.kl_divergence(
          fixed_mean_distribution)  # Shape [B].

    # Compute the alpha-weighted KL-penalty and dual losses to adapt the alphas.
    loss_kl_mean, loss_alpha_mean = compute_parametric_kl_penalty_and_dual_loss(
        kl_mean, alpha_mean, self._epsilon_mean)
    loss_kl_stddev, loss_alpha_stddev = (
        compute_parametric_kl_penalty_and_dual_loss(
            kl_stddev, alpha_stddev, self._epsilon_stddev
        )
    )

    # Combine losses.
    loss_kl_penalty = loss_kl_mean + loss_kl_stddev
    loss_dual = loss_alpha_mean + loss_alpha_stddev
    loss = (self._policy_loss_scale * loss_policy +
            self._kl_loss_scale * loss_kl_penalty +
            self._dual_loss_scale * loss_dual)

    # Create statistics.
    pi_stddev = online_action_distribution.distribution.stddev()
    stats = WPOStats(
        # Dual Variables.
        dual_alpha_mean=jnp.mean(alpha_mean),
        dual_alpha_stddev=jnp.mean(alpha_stddev),
        # Losses.
        loss_policy=jnp.mean(loss),
        loss_alpha=jnp.mean(loss_alpha_mean + loss_alpha_stddev),
        # KL measurements.
        kl_mean_rel=jnp.mean(kl_mean, axis=0) / self._epsilon_mean,
        kl_stddev_rel=jnp.mean(kl_stddev, axis=0) / self._epsilon_stddev,
        # Q measurements.
        q_min=jnp.mean(jnp.min(q_values, axis=0)),
        q_max=jnp.mean(jnp.max(q_values, axis=0)),
        # If the policy has stddev, log summary stats for this as well.
        pi_stddev_min=jnp.mean(jnp.min(pi_stddev, axis=-1)),
        pi_stddev_max=jnp.mean(jnp.max(pi_stddev, axis=-1)),
        # Condition number of the diagonal covariance (actually, stddev) matrix.
        pi_stddev_cond=jnp.mean(
            jnp.max(pi_stddev, axis=-1) / jnp.min(pi_stddev, axis=-1)),
    )

    return loss, stats


def compute_wpo_loss(
    sampled_actions: jnp.ndarray,
    q_value_grads: jnp.ndarray,
    online_action_distribution: tfd.Distribution,
    squashing_type: WPOSquashingType,
    is_terminal: jnp.ndarray,
) -> jnp.ndarray:
  """Compute function whose gradient is the WPO gradient.

  Args:
    sampled_actions: samples used in the Monte Carlo integration in the policy
      loss. Expected shape is [N, B, ...], where N is the number of sampled
      actions and B is the number of sampled states.
    q_value_grads: gradient of the action-value function with respect to the
      sampled actions; expected shape is [N, B, A] where A is the dimensionality
      of the action space.
    online_action_distribution: policy to be optimized.
    squashing_type: Enum of different functions available for squashing the
      action-value gradient in the policy loss. Default is identity.
    is_terminal: boolean array indicating whether the state is terminal;
      expects shape [B].

  Returns:
    loss_policy_gradient: the loss that, when differentiated,
      produces the Wasserstein policy gradient.
  """

  log_probs, log_prob_vjp = jax.vjp(online_action_distribution.log_prob,
                                    jax.lax.stop_gradient(sampled_actions))
  log_prob_vjp_grad = log_prob_vjp(jnp.ones_like(log_probs))[0]

  match squashing_type:
    case WPOSquashingType.IDENTITY:
      squash = lambda x: x
    case WPOSquashingType.CBRT:
      squash = jnp.cbrt
    case _:
      raise ValueError(f'Unsupported squashing type: {squashing_type}')

  loss_policy_gradient = -jnp.sum(
      jnp.sum(
          log_prob_vjp_grad * squash(jax.lax.stop_gradient(q_value_grads)),
          axis=-1),
      axis=0)

  # Return mean loss over the batch, masked by whether the state is terminal.
  not_terminal = jnp.logical_not(is_terminal)
  return reduce_mean(loss_policy_gradient, not_terminal)


def compute_parametric_kl_penalty_and_dual_loss(
    kl: jnp.ndarray,
    alpha: jnp.ndarray,
    epsilon: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes the KL cost to be added to the Lagragian and its dual loss.

  The KL cost is simply the alpha-weighted KL divergence and it is added as a
  regularizer to the policy loss. The dual variable alpha itself has a loss that
  can be minimized to adapt the strength of the regularizer to keep the KL
  between consecutive updates at the desired target value of epsilon.

  Args:
    kl: KL divergence between the target and online policies.
    alpha: Lagrange multipliers (dual variables) for the KL constraints.
    epsilon: Desired value for the KL.

  Returns:
    loss_kl: alpha-weighted KL regularization to be added to the policy loss.
    loss_alpha: The Lagrange dual loss minimized to adapt alpha.
  """

  # Compute the mean KL over the batch.
  mean_kl = jnp.mean(kl, axis=0)

  # Compute the regularization.
  loss_kl = jnp.sum(jax.lax.stop_gradient(alpha) * mean_kl)

  # Compute the dual loss.
  loss_alpha = jnp.sum(alpha * (epsilon - jax.lax.stop_gradient(mean_kl)))

  return loss_kl, loss_alpha


def clip_wpo_params(params: WPOParams) -> WPOParams:
  return WPOParams(
      log_alpha_mean=jnp.maximum(_MIN_LOG_ALPHA, params.log_alpha_mean),
      log_alpha_stddev=jnp.maximum(_MIN_LOG_ALPHA, params.log_alpha_stddev))


def natural_gradient_adaptor(
    dist: tfd.MultivariateNormalDiag | tfd.Independent,
) -> tfd.Independent:
  """A natural gradient adaptor for normal distributions.

  Scales the gradients of this distribution with the variance σ^2 for the mean,
  and half the variance σ^2/2 for the standard deviation.  This implements a
  natural gradient with respect to the distribution parameters, because
  the Fisher matrix for that is diagonal with the inverse of those factors. This
  helps with scaling issues in some algorithms.  It does not change the output
  of log p(a), only the gradients.

  Args:
    dist: The distribution to adapt.

  Returns:
    The adapted distribution. The output of log p(a) will be the same as the
    input log p(a), but the gradients will be scaled with the variance, as
    described above.
  """
  natural_mean = jax.tree.map(
      lambda mi, si: jax.lax.stop_gradient(si**2) * mi
      + jax.lax.stop_gradient((1 - si**2) * mi),
      dist.mean(),
      dist.stddev(),
  )
  natural_stddev = jax.tree.map(
      lambda si: jax.lax.stop_gradient(si**2 / 2) * si
      + jax.lax.stop_gradient((1 - si**2 / 2) * si),
      dist.stddev(),
  )
  dist = tfd.Independent(tfd.Normal(natural_mean, natural_stddev))
  return dist


def reduce_mean(
    array: jnp.ndarray, weights: jnp.ndarray | None = None
) -> jnp.ndarray:
  """Computes a weighted mean of a batch of sequences."""
  if weights is None:
    weights = jnp.ones_like(array, dtype=jnp.float32)
  # Compute weighted mean along the time dimension.
  return weighted_mean(array, weights, axis=0)


def weighted_mean(
    array: jnp.ndarray,
    mask: jnp.ndarray,
    axis: int | tuple[int, ...] | None = None,
) -> jax.Array:
  """Mean of an array with elements weighted by mask.

  Calculates sum_i array_i mask_i / sum_i mask_i along an axis. By convention
  if sum_i mask_i is 0 then the result is 0.

  Args:
    array: Arbitrary array.
    mask: Array of same shape as array.
    axis: One or more axes to average over. None means all axes.

  Returns:
    An array with the all axes in `axis` reduced.
  """
  if array.shape != mask.shape:
    raise ValueError('"array" and "mask" must have the same shape but are '
                     f'{array.shape} and {mask.shape}.')

  summed = jnp.sum(array * mask, axis=axis)
  counts = jnp.sum(mask, axis=axis)
  # Protect against https://github.com/google/jax/issues/1052 by directly
  # setting summed to 0 where counts is zero.
  summed = jnp.where(jnp.equal(counts, 0.), jnp.zeros_like(summed), summed)
  out = jnp.where(jnp.equal(counts, 0.), summed, summed / counts)
  return out
