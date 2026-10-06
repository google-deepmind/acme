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

"""Implements the multi-objective MPO (MO-MPO) loss.

This loss was proposed in (Abdolmaleki, Huang et al., 2020).

The loss is implemented as a Sonnet module rather than a function so that it
can hold its own dual variables, as instances of `tf.Variable`, which it creates
the first time the module is called.

Tensor shapes are annotated, where helpful, as follow:
  B: batch size,
  N: number of sampled actions, see MO-MPO paper for more details,
  D: dimensionality of the action space,
  K: number of objectives.

(Abdolmaleki, Huang et al., 2020): https://arxiv.org/pdf/2005.07513.pdf
"""

import dataclasses
from typing import Dict, Sequence, Tuple, Union

from acme.tf.losses import mpo
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

_MPO_FLOAT_EPSILON = 1e-8


@dataclasses.dataclass
class KLConstraint:
  """Defines a per-objective policy improvement step constraint for MO-MPO."""

  name: str
  value: float

  def __post_init__(self):
    if self.value < 0:
      raise ValueError("KL constraint epsilon must be non-negative.")


class MultiObjectiveMPO(snt.Module):
  """Multi-objective MPO loss with decoupled KL constraints.

  This implementation of the MO-MPO loss is based on the approach proposed in
  (Abdolmaleki, Huang et al., 2020). The following features are included as
  options:
  - Satisfying the KL-constraint on a per-dimension basis (on by default)

  (Abdolmaleki, Huang et al., 2020): https://arxiv.org/pdf/2005.07513.pdf
  """

  def __init__(self,
               epsilons: Sequence[KLConstraint],
               epsilon_mean: float,
               epsilon_stddev: float,
               init_log_temperature: float,
               init_log_alpha_mean: float,
               init_log_alpha_stddev: float,
               per_dim_constraining: bool = True,
               name: str = "MOMPO"):
    """Initialize and configure the MPO loss.

    Args:
      epsilons: per-objective KL constraints on the non-parametric auxiliary
        policy, the one associated with the dual variables called temperature;
        expected length K.
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
      name: a name for the module, passed directly to snt.Module.

    """
    super().__init__(name=name)

    # MO-MPO constraint thresholds.
    self._epsilons = tf.constant([x.value for x in epsilons])
    self._epsilon_mean = tf.constant(epsilon_mean)
    self._epsilon_stddev = tf.constant(epsilon_stddev)

    # Initial values for the constraints' dual variables.
    self._init_log_temperature = init_log_temperature
    self._init_log_alpha_mean = init_log_alpha_mean
    self._init_log_alpha_stddev = init_log_alpha_stddev

    # Whether to ensure per-dimension KL constraint satisfication.
    self._per_dim_constraining = per_dim_constraining

    # Remember the number of objectives
    self._num_objectives = len(epsilons)  # K = number of objectives
    self._objective_names = [x.name for x in epsilons]

    # Make sure there are no duplicate objective names
    if len(self._objective_names) != len(set(self._objective_names)):
      raise ValueError("Duplicate objective names are not allowed.")

  @property
  def objective_names(self):
    return self._objective_names

  @snt.once
  def create_dual_variables_once(self, shape: tf.TensorShape, dtype: tf.DType):
    """Creates the dual variables the first time the loss module is called."""

    # Create the dual variables.
    self._log_temperature = tf.Variable(
        initial_value=[self._init_log_temperature] * self._num_objectives,
        dtype=dtype,
        name="log_temperature",
        shape=(self._num_objectives,))
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
    self._epsilons = tf.cast(self._epsilons, dtype)
    self._epsilon_mean = tf.cast(self._epsilon_mean, dtype)
    self._epsilon_stddev = tf.cast(self._epsilon_stddev, dtype)

  def __call__(
      self,
      online_action_distribution: Union[tfd.MultivariateNormalDiag,
                                        tfd.Independent],
      target_action_distribution: Union[tfd.MultivariateNormalDiag,
                                        tfd.Independent],
      actions: tf.Tensor,  # Shape [N, B, D].
      q_values: tf.Tensor,  # Shape [N, B, K].
  ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    """Computes the decoupled MO-MPO loss.

    Args:
      online_action_distribution: online distribution returned by the online
        policy network; expects batch_dims of [B] and event_dims of [D].
      target_action_distribution: target distribution returned by the target
        policy network; expects same shapes as online distribution.
      actions: actions sampled from the target policy; expects shape [N, B, D].
      q_values: Q-values associated with each action; expects shape [N, B, K].

    Returns:
      Loss, combining the policy loss, KL penalty, and dual losses required to
        adapt the dual variables.
      Stats, for diagnostics and tracking performance.
    """

    # Make sure the Q-values are per-objective
    q_values.get_shape().assert_has_rank(3)
    if q_values.get_shape()[-1] != self._num_objectives:
      raise ValueError("Q-values do not match expected number of objectives.")

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
        q_values, self._epsilons, temperature)  # Shapes [N, B, K] and [1, K].
    normalized_weights_sum = tf.reduce_sum(normalized_weights, axis=-1)
    loss_temperature_mean = tf.reduce_mean(loss_temperature)

    # Only needed for diagnostics: Compute estimated actualized KL between the
    # non-parametric and current target policies.
    kl_nonparametric = mpo.compute_nonparametric_kl_from_normalized_weights(
        normalized_weights)

    # Decompose the online policy into fixed-mean & fixed-stddev distributions.
    # This has been documented as having better performance in bandit settings,
    # see e.g. https://arxiv.org/pdf/1812.02256.pdf.
    fixed_stddev_distribution = tfd.Independent(
        tfd.Normal(loc=online_mean, scale=target_scale))
    fixed_mean_distribution = tfd.Independent(
        tfd.Normal(loc=target_mean, scale=online_scale))

    # Compute the decomposed policy losses.
    loss_policy_mean = mpo.compute_cross_entropy_loss(
        actions, normalized_weights_sum, fixed_stddev_distribution)
    loss_policy_stddev = mpo.compute_cross_entropy_loss(
        actions, normalized_weights_sum, fixed_mean_distribution)

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
    loss_kl_mean, loss_alpha_mean = mpo.compute_parametric_kl_penalty_and_dual_loss(
        kl_mean, alpha_mean, self._epsilon_mean)
    loss_kl_stddev, loss_alpha_stddev = mpo.compute_parametric_kl_penalty_and_dual_loss(
        kl_stddev, alpha_stddev, self._epsilon_stddev)

    # Combine losses.
    loss_policy = loss_policy_mean + loss_policy_stddev
    loss_kl_penalty = loss_kl_mean + loss_kl_stddev
    loss_dual = loss_alpha_mean + loss_alpha_stddev + loss_temperature_mean
    loss = loss_policy + loss_kl_penalty + loss_dual

    stats = {}
    # Dual Variables.
    stats["dual_alpha_mean"] = tf.reduce_mean(alpha_mean)
    stats["dual_alpha_stddev"] = tf.reduce_mean(alpha_stddev)
    # Losses.
    stats["loss_policy"] = tf.reduce_mean(loss)
    stats["loss_alpha"] = tf.reduce_mean(loss_alpha_mean + loss_alpha_stddev)
    # KL measurements.
    stats["kl_mean_rel"] = tf.reduce_mean(kl_mean, axis=0) / self._epsilon_mean
    stats["kl_stddev_rel"] = tf.reduce_mean(
        kl_stddev, axis=0) / self._epsilon_stddev
    # If the policy has standard deviation, log summary stats for this as well.
    pi_stddev = online_action_distribution.distribution.stddev()
    stats["pi_stddev_min"] = tf.reduce_mean(tf.reduce_min(pi_stddev, axis=-1))
    stats["pi_stddev_max"] = tf.reduce_mean(tf.reduce_max(pi_stddev, axis=-1))

    # Condition number of the diagonal covariance (actually, stddev) matrix.
    stats["pi_stddev_cond"] = tf.reduce_mean(
        tf.reduce_max(pi_stddev, axis=-1) / tf.reduce_min(pi_stddev, axis=-1))

    # Log per-objective values.
    for i, name in enumerate(self._objective_names):
      stats["{}_dual_temperature".format(name)] = temperature[i]
      stats["{}_loss_temperature".format(name)] = loss_temperature[i]
      stats["{}_kl_q_rel".format(name)] = tf.reduce_mean(
          kl_nonparametric[:, i]) / self._epsilons[i]

      # Q measurements.
      stats["{}_q_min".format(name)] = tf.reduce_mean(tf.reduce_min(
          q_values, axis=0)[:, i])
      stats["{}_q_mean".format(name)] = tf.reduce_mean(tf.reduce_mean(
          q_values, axis=0)[:, i])
      stats["{}_q_max".format(name)] = tf.reduce_mean(tf.reduce_max(
          q_values, axis=0)[:, i])

    return loss, stats


def compute_weights_and_temperature_loss(
    q_values: tf.Tensor,
    epsilons: tf.Tensor,
    temperature: tf.Variable,
) -> Tuple[tf.Tensor, tf.Tensor]:
  """Computes normalized importance weights for the policy optimization.

  Args:
    q_values: Q-values associated with the actions sampled from the target
      policy; expected shape [N, B, K].
    epsilons: Desired per-objective constraints on the KL between the target
      and non-parametric policies; expected shape [K].
    temperature: Per-objective scalar used to temper the Q-values before
      computing normalized importance weights from them; expected shape [K].
      This is really the Lagrange dual variable in the constrained optimization
      problem, the solution of which is the non-parametric policy targeted by
      the policy loss.

  Returns:
    Normalized importance weights, used for policy optimization; shape [N,B,K].
    Temperature loss, used to adapt the temperature; shape [1, K].
  """

  # Temper the given Q-values using the current temperature.
  tempered_q_values = tf.stop_gradient(q_values) / temperature[None, None, :]

  # Compute the normalized importance weights used to compute expectations with
  # respect to the non-parametric policy.
  normalized_weights = tf.nn.softmax(tempered_q_values, axis=0)
  normalized_weights = tf.stop_gradient(normalized_weights)

  # Compute the temperature loss (dual of the E-step optimization problem).
  q_logsumexp = tf.reduce_logsumexp(tempered_q_values, axis=0)
  log_num_actions = tf.math.log(tf.cast(q_values.shape[0], tf.float32))
  loss_temperature = (
      epsilons + tf.reduce_mean(q_logsumexp, axis=0) - log_num_actions)
  loss_temperature = temperature * loss_temperature

  return normalized_weights, loss_temperature
