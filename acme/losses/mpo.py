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

"""Implements the MPO losses.

The MPO loss is implemented as a Sonnet module rather than a function so that it
can hold its own dual variables, as instances of `tf.Variable`, which it creates
the first time the module is called.

Tensor shapes are annotated, where helpful, as follow:
  B: batch size,
  N: number of sampled actions, see MPO paper for more details,
  D: dimensionality of the action space.
"""

from typing import Dict, Tuple, Union

import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

_MPO_FLOAT_EPSILON = 1e-8


class MPO(snt.Module):
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
               epsilon_penalty: float = 0.001,
               name: str = "MPO"):
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
      init_log_alpha_mean: initial value for the alpha_mean in log-space, note
        a softplus (rather than an exp) will be used to transform this.
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
      name: a name for the module, passed directly to snt.Module.

    """
    super().__init__(name=name)

    # MPO constrain thresholds.
    self._epsilon = tf.constant(epsilon)
    self._epsilon_mean = tf.constant(epsilon_mean)
    self._epsilon_stddev = tf.constant(epsilon_stddev)

    # Initial values for the constraints' dual variables.
    self._init_log_temperature = init_log_temperature
    self._init_log_alpha_mean = init_log_alpha_mean
    self._init_log_alpha_stddev = init_log_alpha_stddev

    # Whether to penalize out-of-bound actions via MO-MPO and its corresponding
    # constraint threshold.
    self._action_penalization = action_penalization
    self._epsilon_penalty = tf.constant(epsilon_penalty)

    # Whether to ensure per-dimension KL constraint satisfication.
    self._per_dim_constraining = per_dim_constraining

  @snt.once
  def create_dual_variables_once(self, shape: tf.TensorShape, dtype: tf.DType):
    """Creates the dual variables the first time the loss module is called."""

    # Create the dual variables.
    self._log_temperature = tf.Variable(
        initial_value=[self._init_log_temperature],
        dtype=dtype,
        name="log_temperature",
        shape=(1,))
    self._log_alpha_mean = tf.Variable(
        initial_value=tf.fill(shape, self._init_log_alpha_mean),
        dtype=dtype,
        name="log_alpha_mean",
        shape=shape)
    self._log_alpha_stddev = tf.Variable(
        initial_value=tf.fill(shape, self._init_log_alpha_stddev),
        dtype=dtype,
        name="log_alpha_stddev",
        shape=shape)

    # Cast constraint thresholds to the expected dtype.
    self._epsilon = tf.cast(self._epsilon, dtype)
    self._epsilon_mean = tf.cast(self._epsilon_mean, dtype)
    self._epsilon_stddev = tf.cast(self._epsilon_stddev, dtype)

    # Maybe create the action penalization dual variable.
    if self._action_penalization:
      self._epsilon_penalty = tf.cast(self._epsilon_penalty, dtype)
      self._log_penalty_temperature = tf.Variable(
          initial_value=[self._init_log_temperature],
          dtype=dtype,
          name="log_penalty_temperature",
          shape=(1,))

  def __call__(
      self,
      online_action_distribution: Union[tfd.MultivariateNormalDiag,
                                        tfd.Independent],
      target_action_distribution: Union[tfd.MultivariateNormalDiag,
                                        tfd.Independent],
      actions: tf.Tensor,  # Shape [N, B, D].
      q_values: tf.Tensor,  # Shape [N, B].
  ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    """Computes the decoupled MPO loss.

    Args:
      online_action_distribution: online distribution returned by the online
        policy network; expects batch_dims of [N, B] and event_dims of [D].
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

    # Infer the shape and dtype of dual variables.
    scalar_dtype = q_values.dtype
    if self._per_dim_constraining:
      dual_variable_shape = target_action_distribution.distribution.kl_divergence(
          online_action_distribution.distribution).shape[1:]  # Should be [D].
    else:
      dual_variable_shape = target_action_distribution.kl_divergence(
          online_action_distribution).shape[1:]  # Should be [1].

    # Create dual variables for the KL constraints; only happens the first call.
    self.create_dual_variables_once(dual_variable_shape, scalar_dtype)

    # Project dual variables to ensure they stay positive.
    min_log_temperature = tf.constant(-18.0, scalar_dtype)
    min_log_alpha = tf.constant(-18.0, scalar_dtype)
    self._log_temperature.assign(
        tf.maximum(min_log_temperature, self._log_temperature))
    self._log_alpha_mean.assign(tf.maximum(min_log_alpha, self._log_alpha_mean))
    self._log_alpha_stddev.assign(
        tf.maximum(min_log_alpha, self._log_alpha_stddev))

    # Transform dual variables from log-space.
    # Note: using softplus instead of exponential for numerical stability.
    temperature = tf.math.softplus(self._log_temperature) + _MPO_FLOAT_EPSILON
    alpha_mean = tf.math.softplus(self._log_alpha_mean) + _MPO_FLOAT_EPSILON
    alpha_stddev = tf.math.softplus(self._log_alpha_stddev) + _MPO_FLOAT_EPSILON

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
      # Project and transform action penalization temperature.
      self._log_penalty_temperature.assign(
          tf.maximum(min_log_temperature, self._log_penalty_temperature))
      penalty_temperature = tf.math.softplus(
          self._log_penalty_temperature) + _MPO_FLOAT_EPSILON

      # Compute action penalization cost.
      # Note: the cost is zero in [-1, 1] and quadratic beyond.
      diff_out_of_bound = actions - tf.clip_by_value(actions, -1.0, 1.0)
      cost_out_of_bound = -tf.norm(diff_out_of_bound, axis=-1)

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
    loss_policy_mean = compute_cross_entropy_loss(
        actions, normalized_weights, fixed_stddev_distribution)
    loss_policy_stddev = compute_cross_entropy_loss(
        actions, normalized_weights, fixed_mean_distribution)

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

    stats = {}
    # Dual Variables.
    stats["dual_alpha_mean"] = tf.reduce_mean(alpha_mean)
    stats["dual_alpha_stddev"] = tf.reduce_mean(alpha_stddev)
    stats["dual_temperature"] = tf.reduce_mean(temperature)
    # Losses.
    stats["loss_policy"] = tf.reduce_mean(loss)
    stats["loss_alpha"] = tf.reduce_mean(loss_alpha_mean + loss_alpha_stddev)
    stats["loss_temperature"] = tf.reduce_mean(loss_temperature)
    # KL measurements.
    stats["kl_q_rel"] = tf.reduce_mean(kl_nonparametric) / self._epsilon

    if self._action_penalization:
      stats["penalty_kl_q_rel"] = tf.reduce_mean(
          penalty_kl_nonparametric) / self._epsilon_penalty

    stats["kl_mean_rel"] = tf.reduce_mean(kl_mean, axis=0) / self._epsilon_mean
    stats["kl_stddev_rel"] = tf.reduce_mean(
        kl_stddev, axis=0) / self._epsilon_stddev
    # Q measurements.
    stats["q_min"] = tf.reduce_mean(tf.reduce_min(q_values, axis=0))
    stats["q_max"] = tf.reduce_mean(tf.reduce_max(q_values, axis=0))
    # If the policy has standard deviation, log summary stats for this as well.
    pi_stddev = online_action_distribution.distribution.stddev()
    stats["pi_stddev_min"] = tf.reduce_mean(tf.reduce_min(pi_stddev, axis=-1))
    stats["pi_stddev_max"] = tf.reduce_mean(tf.reduce_max(pi_stddev, axis=-1))
    # Condition number of the diagonal covariance (actually, stddev) matrix.
    stats["pi_stddev_cond"] = tf.reduce_mean(
        tf.reduce_max(pi_stddev, axis=-1) / tf.reduce_min(pi_stddev, axis=-1))

    return loss, stats


def compute_weights_and_temperature_loss(
    q_values: tf.Tensor,
    epsilon: float,
    temperature: tf.Variable,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Computes normalized importance weights for the policy optimization.

  Args:
    q_values: Q-values associated with the actions sampled from the target
      policy; expected shape [N, B].
    epsilon: Desired constraint on the KL between the target and non-parametric
      policies.
    temperature: Scalar used to temper the Q-values before computing normalized
      importance weights from them. This is really the Lagrange dual variable
      in the constrained optimization problem, the solution of which is the
      non-parametric policy targeted by the policy loss.

  Returns:
    Normalized importance weights, used for policy optimization.
    Temperature loss, used to adapt the temperature.
  """

  # Temper the given Q-values using the current temperature.
  tempered_q_values = tf.stop_gradient(q_values) / temperature

  # Compute the normalized importance weights used to compute expectations with
  # respect to the non-parametric policy.
  normalized_weights = tf.nn.softmax(tempered_q_values, axis=0)
  normalized_weights = tf.stop_gradient(normalized_weights)

  # Compute the temperature loss (dual of the E-step optimization problem).
  q_logsumexp = tf.reduce_logsumexp(tempered_q_values, axis=0)
  log_num_actions = tf.math.log(tf.cast(q_values.shape[0], tf.float32))
  loss_temperature = epsilon + tf.reduce_mean(q_logsumexp) - log_num_actions
  loss_temperature = temperature * loss_temperature

  return normalized_weights, loss_temperature


def compute_nonparametric_kl_from_normalized_weights(
    normalized_weights: tf.Tensor) -> tf.Tensor:
  """Estimate the actualized KL between the non-parametric and target policies."""

  # Compute integrand.
  num_action_samples = tf.cast(normalized_weights.shape[0], tf.float32)
  integrand = tf.math.log(num_action_samples * normalized_weights + 1e-8)

  # Return the expectation with respect to the non-parametric policy.
  return tf.reduce_sum(normalized_weights * integrand, axis=0)


def compute_cross_entropy_loss(
    sampled_actions: tf.Tensor,
    normalized_weights: tf.Tensor,
    online_action_distribution: tfp.distributions.Distribution,
) -> tf.Tensor:
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
  loss_policy_gradient = -tf.reduce_sum(log_prob * normalized_weights, axis=0)

  # Return the mean loss over the batch of states.
  return tf.reduce_mean(loss_policy_gradient, axis=0)


def compute_parametric_kl_penalty_and_dual_loss(
    kl: tf.Tensor,
    alpha: tf.Variable,
    epsilon: float,
) -> Tuple[tf.Tensor, tf.Tensor]:
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
  mean_kl = tf.reduce_mean(kl, axis=0)

  # Compute the regularization.
  loss_kl = tf.reduce_sum(tf.stop_gradient(alpha) * mean_kl)

  # Compute the dual loss.
  loss_alpha = tf.reduce_sum(alpha * (epsilon - tf.stop_gradient(mean_kl)))

  return loss_kl, loss_alpha
