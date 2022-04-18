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

"""Implements the MPO loss.

The MPO loss uses MPOParams, which can be initialized using init_params,
to track the temperature and the dual variables.

Tensor shapes are annotated, where helpful, as follow:
  B: batch size,
  N: number of sampled actions, see MPO paper for more details,
  D: dimensionality of the action space.
"""

from typing import NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import tensorflow_probability

tfp = tensorflow_probability.substrates.jax
tfd = tensorflow_probability.substrates.jax.distributions

_MPO_FLOAT_EPSILON = 1e-8
_MIN_LOG_TEMPERATURE = -18.0
_MIN_LOG_ALPHA = -18.0

Shape = Tuple[int]
DType = type(jnp.float32)  # _ScalarMeta, a private type.


class MPOParams(NamedTuple):
  """NamedTuple to store trainable loss parameters."""
  log_temperature: jnp.ndarray
  log_alpha_mean: jnp.ndarray
  log_alpha_stddev: jnp.ndarray
  log_penalty_temperature: Optional[jnp.ndarray] = None


class MPOStats(NamedTuple):
  """NamedTuple to store loss statistics."""
  dual_alpha_mean: float
  dual_alpha_stddev: float
  dual_temperature: float

  loss_policy: float
  loss_alpha: float
  loss_temperature: float
  kl_q_rel: float

  kl_mean_rel: float
  kl_stddev_rel: float

  q_min: float
  q_max: float

  pi_stddev_min: float
  pi_stddev_max: float
  pi_stddev_cond: float

  penalty_kl_q_rel: Optional[float] = None


class MPO:
  """MPO loss with decoupled KL constraints as in (Abdolmaleki et al., 2018).

  This implementation of the MPO loss includes the following features, as
  options:
  - Satisfying the KL-constraint on a per-dimension basis (on by default);
  - Penalizing actions that fall outside of [-1, 1] (on by default) as a
      special case of multi-objective MPO (MO-MPO; Abdolmaleki et al., 2020).
  For best results on the control suite, keep both of these on.

  (Abdolmaleki et al., 2018): https://arxiv.org/pdf/1812.02256.pdf
  (Abdolmaleki et al., 2020): https://arxiv.org/pdf/2005.07513.pdf
  """

  def __init__(self,
               epsilon: float,
               epsilon_mean: float,
               epsilon_stddev: float,
               init_log_temperature: float,
               init_log_alpha_mean: float,
               init_log_alpha_stddev: float,
               per_dim_constraining: bool = True,
               action_penalization: bool = True,
               epsilon_penalty: float = 0.001):
    """Initialize and configure the MPO loss.

    Args:
      epsilon: KL constraint on the non-parametric auxiliary policy, the one
        associated with the dual variable called temperature.
      epsilon_mean: KL constraint on the mean of the Gaussian policy, the one
        associated with the dual variable called alpha_mean.
      epsilon_stddev: KL constraint on the stddev of the Gaussian policy, the
        one associated with the dual variable called alpha_mean.
      init_log_temperature: initial value for the temperature in log-space, note
        a softplus (rather than an exp) will be used to transform this.
      init_log_alpha_mean: initial value for the alpha_mean in log-space, note a
        softplus (rather than an exp) will be used to transform this.
      init_log_alpha_stddev: initial value for the alpha_stddev in log-space,
        note a softplus (rather than an exp) will be used to transform this.
      per_dim_constraining: whether to enforce the KL constraint on each
        dimension independently; this is the default. Otherwise the overall KL
        is constrained, which allows some dimensions to change more at the
        expense of others staying put.
      action_penalization: whether to use a KL constraint to penalize actions
        via the MO-MPO algorithm.
      epsilon_penalty: KL constraint on the probability of violating the action
        constraint.
    """

    # MPO constrain thresholds.
    self._epsilon = epsilon
    self._epsilon_mean = epsilon_mean
    self._epsilon_stddev = epsilon_stddev

    # Initial values for the constraints' dual variables.
    self._init_log_temperature = init_log_temperature
    self._init_log_alpha_mean = init_log_alpha_mean
    self._init_log_alpha_stddev = init_log_alpha_stddev

    # Whether to penalize out-of-bound actions via MO-MPO and its corresponding
    # constraint threshold.
    self._action_penalization = action_penalization
    self._epsilon_penalty = epsilon_penalty

    # Whether to ensure per-dimension KL constraint satisfication.
    self._per_dim_constraining = per_dim_constraining

  @property
  def per_dim_constraining(self):
    return self._per_dim_constraining

  def init_params(self, action_dim: int, dtype: DType = jnp.float32):
    """Creates an initial set of parameters."""

    if self._per_dim_constraining:
      dual_variable_shape = [action_dim]
    else:
      dual_variable_shape = [1]

    log_temperature = jnp.full([1], self._init_log_temperature, dtype=dtype)

    log_alpha_mean = jnp.full(
        dual_variable_shape, self._init_log_alpha_mean, dtype=dtype)

    log_alpha_stddev = jnp.full(
        dual_variable_shape, self._init_log_alpha_stddev, dtype=dtype)

    if self._action_penalization:
      log_penalty_temperature = jnp.full([1],
                                         self._init_log_temperature,
                                         dtype=dtype)
    else:
      log_penalty_temperature = None

    return MPOParams(
        log_temperature=log_temperature,
        log_alpha_mean=log_alpha_mean,
        log_alpha_stddev=log_alpha_stddev,
        log_penalty_temperature=log_penalty_temperature)

  def __call__(
      self,
      params: MPOParams,
      online_action_distribution: Union[tfd.MultivariateNormalDiag,
                                        tfd.Independent],
      target_action_distribution: Union[tfd.MultivariateNormalDiag,
                                        tfd.Independent],
      actions: jnp.ndarray,  # Shape [N, B, D].
      q_values: jnp.ndarray,  # Shape [N, B].
  ) -> Tuple[jnp.ndarray, MPOStats]:
    """Computes the decoupled MPO loss.

    Args:
      params: parameters tracking the temperature and the dual variables.
      online_action_distribution: online distribution returned by the online
        policy network; expects batch_dims of [B] and event_dims of [D].
      target_action_distribution: target distribution returned by the target
        policy network; expects same shapes as online distribution.
      actions: actions sampled from the target policy; expects shape [N, B, D].
      q_values: Q-values associated with each action; expects shape [N, B].

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
    temperature = jax.nn.softplus(params.log_temperature) + _MPO_FLOAT_EPSILON
    alpha_mean = jax.nn.softplus(params.log_alpha_mean) + _MPO_FLOAT_EPSILON
    alpha_stddev = jax.nn.softplus(params.log_alpha_stddev) + _MPO_FLOAT_EPSILON

    # Get online and target means and stddevs in preparation for decomposition.
    online_mean = online_action_distribution.distribution.mean()
    online_scale = online_action_distribution.distribution.stddev()
    target_mean = target_action_distribution.distribution.mean()
    target_scale = target_action_distribution.distribution.stddev()

    # Compute normalized importance weights, used to compute expectations with
    # respect to the non-parametric policy; and the temperature loss, used to
    # adapt the tempering of Q-values.
    normalized_weights, loss_temperature = compute_weights_and_temperature_loss(
        q_values, self._epsilon, temperature)

    # Only needed for diagnostics: Compute estimated actualized KL between the
    # non-parametric and current target policies.
    kl_nonparametric = compute_nonparametric_kl_from_normalized_weights(
        normalized_weights)

    if self._action_penalization:
      # Transform action penalization temperature.
      penalty_temperature = jax.nn.softplus(
          params.log_penalty_temperature) + _MPO_FLOAT_EPSILON

      # Compute action penalization cost.
      # Note: the cost is zero in [-1, 1] and quadratic beyond.
      diff_out_of_bound = actions - jnp.clip(actions, -1.0, 1.0)
      cost_out_of_bound = -jnp.linalg.norm(diff_out_of_bound, axis=-1)

      penalty_normalized_weights, loss_penalty_temperature = compute_weights_and_temperature_loss(
          cost_out_of_bound, self._epsilon_penalty, penalty_temperature)

      # Only needed for diagnostics: Compute estimated actualized KL between the
      # non-parametric and current target policies.
      penalty_kl_nonparametric = compute_nonparametric_kl_from_normalized_weights(
          penalty_normalized_weights)

      # Combine normalized weights.
      normalized_weights += penalty_normalized_weights
      loss_temperature += loss_penalty_temperature

    # Decompose the online policy into fixed-mean & fixed-stddev distributions.
    # This has been documented as having better performance in bandit settings,
    # see e.g. https://arxiv.org/pdf/1812.02256.pdf.
    fixed_stddev_distribution = tfd.Independent(
        tfd.Normal(loc=online_mean, scale=target_scale))
    fixed_mean_distribution = tfd.Independent(
        tfd.Normal(loc=target_mean, scale=online_scale))

    # Compute the decomposed policy losses.
    loss_policy_mean = compute_cross_entropy_loss(actions, normalized_weights,
                                                  fixed_stddev_distribution)
    loss_policy_stddev = compute_cross_entropy_loss(actions, normalized_weights,
                                                    fixed_mean_distribution)

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
    loss_kl_stddev, loss_alpha_stddev = compute_parametric_kl_penalty_and_dual_loss(
        kl_stddev, alpha_stddev, self._epsilon_stddev)

    # Combine losses.
    loss_policy = loss_policy_mean + loss_policy_stddev
    loss_kl_penalty = loss_kl_mean + loss_kl_stddev
    loss_dual = loss_alpha_mean + loss_alpha_stddev + loss_temperature
    loss = loss_policy + loss_kl_penalty + loss_dual

    # Create statistics.
    pi_stddev = online_action_distribution.distribution.stddev()
    stats = MPOStats(
        # Dual Variables.
        dual_alpha_mean=jnp.mean(alpha_mean),
        dual_alpha_stddev=jnp.mean(alpha_stddev),
        dual_temperature=jnp.mean(temperature),
        # Losses.
        loss_policy=jnp.mean(loss),
        loss_alpha=jnp.mean(loss_alpha_mean + loss_alpha_stddev),
        loss_temperature=jnp.mean(loss_temperature),
        # KL measurements.
        kl_q_rel=jnp.mean(kl_nonparametric) / self._epsilon,
        penalty_kl_q_rel=((jnp.mean(penalty_kl_nonparametric) /
                           self._epsilon_penalty)
                          if self._action_penalization else None),
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


def compute_weights_and_temperature_loss(
    q_values: jnp.ndarray,
    epsilon: float,
    temperature: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Computes normalized importance weights for the policy optimization.

  Args:
    q_values: Q-values associated with the actions sampled from the target
      policy; expected shape [N, B].
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

  # Compute the normalized importance weights used to compute expectations with
  # respect to the non-parametric policy.
  normalized_weights = jax.nn.softmax(tempered_q_values, axis=0)
  normalized_weights = jax.lax.stop_gradient(normalized_weights)

  # Compute the temperature loss (dual of the E-step optimization problem).
  q_logsumexp = jax.scipy.special.logsumexp(tempered_q_values, axis=0)
  log_num_actions = jnp.log(q_values.shape[0] / 1.)
  loss_temperature = epsilon + jnp.mean(q_logsumexp) - log_num_actions
  loss_temperature = temperature * loss_temperature

  return normalized_weights, loss_temperature


def compute_nonparametric_kl_from_normalized_weights(
    normalized_weights: jnp.ndarray) -> jnp.ndarray:
  """Estimate the actualized KL between the non-parametric and target policies."""

  # Compute integrand.
  num_action_samples = normalized_weights.shape[0] / 1.
  integrand = jnp.log(num_action_samples * normalized_weights + 1e-8)

  # Return the expectation with respect to the non-parametric policy.
  return jnp.sum(normalized_weights * integrand, axis=0)


def compute_cross_entropy_loss(
    sampled_actions: jnp.ndarray,
    normalized_weights: jnp.ndarray,
    online_action_distribution: tfd.Distribution,
) -> jnp.ndarray:
  """Compute cross-entropy online and the reweighted target policy.

  Args:
    sampled_actions: samples used in the Monte Carlo integration in the policy
      loss. Expected shape is [N, B, ...], where N is the number of sampled
      actions and B is the number of sampled states.
    normalized_weights: target policy multiplied by the exponentiated Q values
      and normalized; expected shape is [N, B].
    online_action_distribution: policy to be optimized.

  Returns:
    loss_policy_gradient: the cross-entropy loss that, when differentiated,
      produces the policy gradient.
  """

  # Compute the M-step loss.
  log_prob = online_action_distribution.log_prob(sampled_actions)

  # Compute the weighted average log-prob using the normalized weights.
  loss_policy_gradient = -jnp.sum(log_prob * normalized_weights, axis=0)

  # Return the mean loss over the batch of states.
  return jnp.mean(loss_policy_gradient, axis=0)


def compute_parametric_kl_penalty_and_dual_loss(
    kl: jnp.ndarray,
    alpha: jnp.ndarray,
    epsilon: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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


def clip_mpo_params(params: MPOParams, per_dim_constraining: bool) -> MPOParams:
  clipped_params = MPOParams(
      log_temperature=jnp.maximum(_MIN_LOG_TEMPERATURE, params.log_temperature),
      log_alpha_mean=jnp.maximum(_MIN_LOG_ALPHA, params.log_alpha_mean),
      log_alpha_stddev=jnp.maximum(_MIN_LOG_ALPHA, params.log_alpha_stddev))
  if not per_dim_constraining:
    return clipped_params
  else:
    return clipped_params._replace(
        log_penalty_temperature=jnp.maximum(_MIN_LOG_TEMPERATURE,
                                            params.log_penalty_temperature))
