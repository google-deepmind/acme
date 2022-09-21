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

"""MPO learner implementation. With MoG/not and continuous/discrete policies."""

import copy
import dataclasses
from typing import Tuple

from acme import types
from acme.adders import reverb as adders
from acme.agents.jax.mpo import categorical_mpo as discrete_losses
from acme.agents.jax.mpo import networks as mpo_networks
from acme.agents.jax.mpo import types as mpo_types
from acme.agents.jax.mpo import utils as mpo_utils
from acme.jax import networks as network_lib
import acme.jax.losses.mpo as continuous_losses
import chex
import jax
import jax.numpy as jnp
import rlax


def softmax_cross_entropy(
    logits: chex.Array, target_probs: chex.Array) -> chex.Array:
  """Compute cross entropy loss between logits and target probabilities."""
  chex.assert_equal_shape([target_probs, logits])
  return -jnp.sum(target_probs * jax.nn.log_softmax(logits), axis=-1)


def top1_accuracy_tiebreak(
    logits: chex.Array,
    targets: chex.Array,
    *,
    rng: chex.PRNGKey,
    eps: float = 1e-6) -> chex.Array:
  """Compute the top-1 accuracy with an argmax of targets (random tie-break)."""
  noise = jax.random.uniform(rng, shape=targets.shape,
                             minval=-eps, maxval=eps)
  acc = jnp.argmax(logits, axis=-1) == jnp.argmax(targets + noise, axis=-1)
  return jnp.mean(acc)


class RolloutLoss:
  """A MuZero/Muesli-style loss on the rollouts of the dynamics model."""

  def __init__(
      self,
      dynamics_model: mpo_networks.UnrollableNetwork,
      model_rollout_length: int,
      loss_scales: mpo_types.LossScalesConfig,
      root_policy_loss_config: mpo_types.PolicyLossConfig,
      distributional_loss_fn: mpo_types.DistributionalLossFn,
  ):
    self._dynamics_model = dynamics_model
    self._model_rollout_length = model_rollout_length
    self._loss_scales = loss_scales
    self._distributional_loss_fn = distributional_loss_fn
    self._policy_loss_module = None

    # Create rollout policy loss.
    eps_scale = 1.0
    eps_policy_scale = 10.
    policy_config = copy.deepcopy(dataclasses.asdict(root_policy_loss_config))
    if isinstance(root_policy_loss_config,
                  mpo_types.CategoricalPolicyLossConfig):
      policy_config['epsilon'] = eps_scale * policy_config['epsilon']
      policy_config['epsilon_policy'] = (
          eps_policy_scale * policy_config['epsilon_policy'])
      self._policy_loss_module = discrete_losses.CategoricalMPO(**policy_config)
    elif isinstance(root_policy_loss_config,
                    mpo_types.GaussianPolicyLossConfig):
      policy_config['epsilon'] = eps_scale * policy_config['epsilon']
      policy_config['epsilon_mean'] = (
          eps_policy_scale * policy_config['epsilon_mean'])
      policy_config['epsilon_stddev'] = (
          eps_policy_scale * policy_config['epsilon_stddev'])
      self._policy_loss_module = continuous_losses.MPO(**policy_config)
    else:
      raise ValueError('invalid root_policy_loss_config type '
                       f'{type(root_policy_loss_config)}')

  def _rolling_window(self, x: chex.Array, axis: int = 0) -> chex.Array:
    """A convenient tree-mapped and config'd call to rolling window.

    Stacks R=T-K+1 action slices of length K = model_rollout_length from
    tensor x: [..., 0:K; ...; T-K:T, ...].

    Args:
      x: The tensor to select rolling slices from (along specified axis), with
        shape [..., T, ...] such that T = x.shape[axis].
      axis: The axis to slice from (defaults to 0).

    Returns:
      A tensor containing the stacked slices [0:K, ... T-K:T] from an axis of x
      with shape [..., K, R, ...] for input shape [..., T, ...].
    """
    def rw(y):
      return mpo_utils.rolling_window(
          y, window=self._model_rollout_length, axis=axis, time_major=True)

    return mpo_utils.tree_map_distribution(rw, x)

  def _compute_model_rollout_predictions(
      self, params: mpo_networks.MPONetworkParams,
      state_embeddings: types.NestedArray,
      action_sequence: types.NestedArray) -> mpo_types.ModelOutputs:
    """Roll out the dynamics model for each embedding state."""
    assert self._model_rollout_length > 0
    # Stack the R=T-K+1 action slices of length K: [0:K; ...; T-K:T]; [K, R].
    rollout_actions = self._rolling_window(action_sequence)

    # Create batch of root states (embeddings) s_t for t \in {0, ..., R}.
    num_rollouts = action_sequence.shape[0] - self._model_rollout_length + 1
    root_state = self._dynamics_model.initial_state_fn(
        params.dynamics_model_initial_state, state_embeddings[:num_rollouts])
    # TODO(abef): randomly choose (fewer?) root unroll states, as in Muesli?

    # Roll out K steps forward in time for each root embedding; [K, R, ...].
    # For example, policy_rollout[k, t] is the step-k prediction starting from
    # state s_t (and same for value_rollout and reward_rollout). Thus, for
    # valid values of k, t, and i, policy_rollout[k, t] and
    # policy_rollout[k-i, t+i] share the same target.
    (policy_rollout, value_rollout, reward_rollout,
     embedding_rollout), _ = self._dynamics_model.unroll(
         params.dynamics_model, rollout_actions, root_state)
    # TODO(abef): try using the same params for both the root & rollout heads.

    chex.assert_shape([rollout_actions, embedding_rollout],
                      (self._model_rollout_length, num_rollouts, ...))

    # Create the outputs but drop the rollout that uses action a_{T-1} (and
    # thus contains state s_T) for the policy, value, and embedding because we
    # don't have targets for s_T (but we do know them for the final reward).
    # Also drop the rollout with s_{T-1} for the value because we don't have
    # targets for that either.
    return mpo_types.ModelOutputs(
        policy=policy_rollout[:, :-1],  # [K, R-1, ...]
        value=value_rollout[:, :-2],  # [K, R-2, ...]
        reward=reward_rollout,  # [K, R, ...]
        embedding=embedding_rollout[:, :-1])  # [K, R-1, ...]

  def __call__(
      self, params: mpo_networks.MPONetworkParams,
      dual_params: mpo_types.DualParams, sequence: adders.Step,
      state_embeddings: types.NestedArray, targets: mpo_types.LossTargets,
      key: network_lib.PRNGKey) -> Tuple[jnp.ndarray, mpo_types.LogDict]:

    # Create rollout predictions.
    rollout = self._compute_model_rollout_predictions(
        params=params, state_embeddings=state_embeddings,
        action_sequence=sequence.action)

    # Create rollout target tensors. The rollouts will not contain the policy
    # and value at t=0 because they start after taking the first action in
    # the sequence, so drop those when creating the targets. They will contain
    # the reward at t=0, however, because of how the sequences are stored.
    # Rollout target shapes:
    #   - policy: [T-1, ...] -> [K, R-1, ...],
    #   - a/q_improvement: [N, T-1] -> [N, K, R-1].
    #   - value: [N, Z, T-2] -> [N, Z, K, R-2],
    #   - reward: [T] -> [K, R].
    policy_targets = self._rolling_window(targets.policy[1:])
    a_improvement = self._rolling_window(targets.a_improvement[:, 1:], axis=1)
    q_improvement = self._rolling_window(targets.q_improvement[:, 1:], axis=1)
    value_targets = self._rolling_window(targets.value[..., 1:], axis=-1)
    reward_targets = self._rolling_window(targets.reward)[None, None, ...]
    # TODO(abef): should we re-sample a_ and q_improvement here?

    num_rollouts = sequence.reward.shape[0] - self._model_rollout_length + 1
    num_actions = rollout.policy.logits.shape[-1]  # A
    bc_targets = self._rolling_window(  # [T-1, A] -> [K, R-1, A]
        rlax.one_hot(sequence.action[1:], num_actions))

    # Create the rollout losses.
    def policy_loss_fn(root_idx) -> Tuple[jnp.ndarray, mpo_types.PolicyStats]:
      chex.assert_shape((rollout.policy.logits, policy_targets.logits),
                        (self._model_rollout_length, num_rollouts-1, None))
      chex.assert_shape((a_improvement, q_improvement),
                        (None, self._model_rollout_length, num_rollouts-1))
      return self._policy_loss_module(
          params=dual_params,
          online_action_distribution=rollout.policy[:, root_idx],
          target_action_distribution=policy_targets[:, root_idx],
          actions=a_improvement[..., root_idx],
          q_values=q_improvement[..., root_idx])

    def bc_policy_loss_fn(root_idx) -> Tuple[jnp.ndarray, jnp.ndarray]:
      """Self-behavior-cloning loss (cross entropy on rollout actions)."""
      chex.assert_shape(
          (rollout.policy.logits, bc_targets),
          (self._model_rollout_length, num_rollouts-1, num_actions))
      loss = softmax_cross_entropy(
          rollout.policy.logits[:, root_idx], bc_targets[:, root_idx])
      top1_accuracy = top1_accuracy_tiebreak(
          rollout.policy.logits[:, root_idx], bc_targets[:, root_idx], rng=key)
      return loss, top1_accuracy

    def value_loss_fn(root_idx) -> jnp.ndarray:
      return self._distributional_loss_fn(
          rollout.value[:, root_idx],  # [K, R-2, ...]
          value_targets[..., root_idx])  # [..., K, R-2]

    def reward_loss_fn(root_idx) -> jnp.ndarray:
      return self._distributional_loss_fn(
          rollout.reward[:, root_idx],  # [K, R, ...]
          reward_targets[..., root_idx])  # [..., K, R]

    # Compute each rollout loss by vmapping over the rollouts.
    indices = jnp.arange(num_rollouts)
    mpo_policy_loss, policy_stats = jax.vmap(policy_loss_fn)(indices[:-1])
    bc_policy_loss, bc_policy_acc = jax.vmap(bc_policy_loss_fn)(indices[:-1])
    mpo_policy_loss = jnp.mean(mpo_policy_loss)
    bc_policy_loss = jnp.mean(bc_policy_loss)
    bc_policy_acc = jnp.mean(bc_policy_acc)
    critic_loss = jnp.mean(jax.vmap(value_loss_fn)(indices[:-2]))
    reward_loss = jnp.mean(jax.vmap(reward_loss_fn)(indices))

    # Combine losses.
    policy_loss = (self._loss_scales.rollout.policy * mpo_policy_loss +
                   self._loss_scales.rollout.bc_policy * bc_policy_loss)
    critic_loss = self._loss_scales.rollout.critic * critic_loss
    loss = (self._loss_scales.policy * policy_loss +
            self._loss_scales.critic * critic_loss +
            self._loss_scales.rollout.reward * reward_loss)

    logging_dict = {
        'rollout_mpo_policy_loss': mpo_policy_loss,
        'rollout_bc_policy_loss': bc_policy_loss,
        'rollout_policy_loss': policy_loss,
        'rollout_critic_loss': critic_loss,
        'rollout_reward_loss': reward_loss,
        'rollout_bc_accuracy': bc_policy_acc,
        'rollout_loss': loss,
    }

    logging_dict.update({
        f'policy/rollout/{k}': v for k, v in policy_stats._asdict().items()})

    return loss, logging_dict
