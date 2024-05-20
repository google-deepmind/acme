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

import dataclasses
import functools
import time
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union

from absl import logging
import acme
from acme import specs
from acme import types
from acme.adders import reverb as adders
from acme.agents.jax.mpo import categorical_mpo as discrete_losses
from acme.agents.jax.mpo import networks as mpo_networks
from acme.agents.jax.mpo import rollout_loss
from acme.agents.jax.mpo import types as mpo_types
from acme.agents.jax.mpo import utils as mpo_utils
from acme.jax import networks as network_lib
from acme.jax import types as jax_types
from acme.jax import utils
import acme.jax.losses.mpo as continuous_losses
from acme.utils import counting
from acme.utils import loggers
import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb
import rlax
import tree

_PMAP_AXIS_NAME = 'data'
CriticType = mpo_types.CriticType


class TrainingState(NamedTuple):
  """Contains training state for the learner."""
  params: mpo_networks.MPONetworkParams
  target_params: mpo_networks.MPONetworkParams
  dual_params: mpo_types.DualParams
  opt_state: optax.OptState
  dual_opt_state: optax.OptState
  steps: int
  random_key: jax_types.PRNGKey


def softmax_cross_entropy(
    logits: chex.Array, target_probs: chex.Array) -> chex.Array:
  """Compute cross entropy loss between logits and target probabilities."""
  chex.assert_equal_shape([target_probs, logits])
  return -jnp.sum(target_probs * jax.nn.log_softmax(logits), axis=-1)


def top1_accuracy_tiebreak(logits: chex.Array,
                           targets: chex.Array,
                           *,
                           rng: jax_types.PRNGKey,
                           eps: float = 1e-6) -> chex.Array:
  """Compute the top-1 accuracy with an argmax of targets (random tie-break)."""
  noise = jax.random.uniform(rng, shape=targets.shape,
                             minval=-eps, maxval=eps)
  acc = jnp.argmax(logits, axis=-1) == jnp.argmax(targets + noise, axis=-1)
  return jnp.mean(acc)


class MPOLearner(acme.Learner):
  """MPO learner (discrete or continuous, distributional or not)."""

  _state: TrainingState

  def __init__(  # pytype: disable=annotation-type-mismatch  # numpy-scalars
      self,
      critic_type: CriticType,
      discrete_policy: bool,
      environment_spec: specs.EnvironmentSpec,
      networks: mpo_networks.MPONetworks,
      random_key: jax_types.PRNGKey,
      discount: float,
      num_samples: int,
      iterator: Iterator[reverb.ReplaySample],
      experience_type: mpo_types.ExperienceType,
      loss_scales: mpo_types.LossScalesConfig,
      target_update_period: Optional[int] = 100,
      target_update_rate: Optional[float] = None,
      sgd_steps_per_learner_step: int = 20,
      policy_eval_stochastic: bool = True,
      policy_eval_num_val_samples: int = 128,
      policy_loss_config: Optional[mpo_types.PolicyLossConfig] = None,
      use_online_policy_to_bootstrap: bool = False,
      use_stale_state: bool = False,
      use_retrace: bool = False,
      retrace_lambda: float = 0.95,
      model_rollout_length: int = 0,
      optimizer: Optional[optax.GradientTransformation] = None,
      learning_rate: optax.ScalarOrSchedule = 1e-4,
      dual_optimizer: Optional[optax.GradientTransformation] = None,
      dual_learning_rate: optax.ScalarOrSchedule = 1e-2,
      grad_norm_clip: float = 40.0,
      reward_clip: float = np.float32('inf'),
      value_tx_pair: rlax.TxPair = rlax.IDENTITY_PAIR,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      devices: Optional[Sequence[jax.Device]] = None,
  ):
    self._critic_type = critic_type
    self._discrete_policy = discrete_policy

    process_id = jax.process_index()
    local_devices = jax.local_devices()
    self._devices = devices or local_devices
    logging.info('Learner process id: %s. Devices passed: %s', process_id,
                 devices)
    logging.info('Learner process id: %s. Local devices from JAX API: %s',
                 process_id, local_devices)
    self._local_devices = [d for d in self._devices if d in local_devices]

    # Store networks.
    self._networks = networks

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger

    # Other learner parameters.
    self._discount = discount
    self._num_samples = num_samples
    self._sgd_steps_per_learner_step = sgd_steps_per_learner_step

    self._policy_eval_stochastic = policy_eval_stochastic
    self._policy_eval_num_val_samples = policy_eval_num_val_samples

    self._reward_clip_range = sorted([-reward_clip, reward_clip])
    self._tx_pair = value_tx_pair
    self._loss_scales = loss_scales
    self._use_online_policy_to_bootstrap = use_online_policy_to_bootstrap
    self._model_rollout_length = model_rollout_length

    self._use_retrace = use_retrace
    self._retrace_lambda = retrace_lambda
    if use_retrace and critic_type == CriticType.MIXTURE_OF_GAUSSIANS:
      logging.warning(
          'Warning! Retrace has not been tested with the MoG critic.')
    self._use_stale_state = use_stale_state

    self._experience_type = experience_type
    if isinstance(self._experience_type, mpo_types.FromTransitions):
      # Each n=5-step transition will be converted to a length 2 sequence before
      # being passed to the loss, so we do n=1 step bootstrapping on the
      # resulting sequence to get n=5-step bootstrapping as intended.
      self._n_step_for_sequence_bootstrap = 1
      self._td_lambda = 1.0
    elif isinstance(self._experience_type, mpo_types.FromSequences):
      self._n_step_for_sequence_bootstrap = self._experience_type.n_step
      self._td_lambda = self._experience_type.td_lambda

    # Necessary to track when to update target networks.
    self._target_update_period = target_update_period
    self._target_update_rate = target_update_rate
    # Assert one and only one of target update period or rate is defined.
    if ((target_update_period and target_update_rate) or
        (target_update_period is None and target_update_rate is None)):
      raise ValueError(
          'Exactly one of target_update_{period|rate} must be set.'
          f' Received target_update_period={target_update_period} and'
          f' target_update_rate={target_update_rate}.')

    # Create policy loss.
    if self._discrete_policy:
      policy_loss_config = (
          policy_loss_config or mpo_types.CategoricalPolicyLossConfig())
      self._policy_loss_module = discrete_losses.CategoricalMPO(
          **dataclasses.asdict(policy_loss_config))
    else:
      policy_loss_config = (
          policy_loss_config or mpo_types.GaussianPolicyLossConfig())
      self._policy_loss_module = continuous_losses.MPO(
          **dataclasses.asdict(policy_loss_config))

    self._policy_loss_module.__call__ = jax.named_call(
        self._policy_loss_module.__call__, name='policy_loss')

    # Create the dynamics model rollout loss.
    if model_rollout_length > 0:
      if not discrete_policy and (self._loss_scales.rollout.policy or
                                  self._loss_scales.rollout.bc_policy):
        raise ValueError('Policy rollout losses are only supported in the '
                         'discrete policy case.')
      self._model_rollout_loss_fn = rollout_loss.RolloutLoss(
          dynamics_model=networks.dynamics_model,
          model_rollout_length=model_rollout_length,
          loss_scales=loss_scales,
          distributional_loss_fn=self._distributional_loss)

    # Create optimizers if they aren't given.
    self._optimizer = optimizer or _get_default_optimizer(
        learning_rate, grad_norm_clip
    )
    self._dual_optimizer = dual_optimizer or _get_default_optimizer(
        dual_learning_rate, grad_norm_clip
    )

    self._action_spec = environment_spec.actions

    # Initialize random key for the rest of training.
    random_key, key = jax.random.split(random_key)

    # Initialize network parameters, ignoring the dummy initial state.
    network_params, _ = mpo_networks.init_params(
        self._networks,
        environment_spec,
        key,
        add_batch_dim=True,
        dynamics_rollout_length=self._model_rollout_length)

    # Get action dims (unused in the discrete case).
    dummy_action = utils.zeros_like(environment_spec.actions)
    dummy_action_concat = utils.batch_concat(dummy_action, num_batch_dims=0)

    if isinstance(self._policy_loss_module, discrete_losses.CategoricalMPO):
      self._dual_clip_fn = discrete_losses.clip_categorical_mpo_params
    elif isinstance(self._policy_loss_module, continuous_losses.MPO):
      is_constraining = self._policy_loss_module.per_dim_constraining
      self._dual_clip_fn = lambda dp: continuous_losses.clip_mpo_params(  # pylint: disable=g-long-lambda  # pytype: disable=wrong-arg-types  # numpy-scalars
          dp,
          per_dim_constraining=is_constraining)

    # Create dual parameters. In the discrete case, the action dim is unused.
    dual_params = self._policy_loss_module.init_params(
        action_dim=dummy_action_concat.shape[-1], dtype=jnp.float32)

    # Initialize optimizers.
    opt_state = self._optimizer.init(network_params)
    dual_opt_state = self._dual_optimizer.init(dual_params)

    # Initialise training state (parameters and optimiser state).
    state = TrainingState(
        params=network_params,
        target_params=network_params,
        dual_params=dual_params,
        opt_state=opt_state,
        dual_opt_state=dual_opt_state,
        steps=0,
        random_key=random_key,
    )
    self._state = utils.replicate_in_all_devices(state, self._local_devices)

    # Log how many parameters the network has.
    sizes = tree.map_structure(jnp.size, network_params)._asdict()
    num_params_by_component_str = ' | '.join(
        [f'{key}: {sum(tree.flatten(size))}' for key, size in sizes.items()])
    logging.info('Number of params by network component: %s',
                 num_params_by_component_str)
    logging.info('Total number of params: %d',
                 sum(tree.flatten(sizes.values())))

    # Combine multiple SGD steps and pmap across devices.
    sgd_steps = utils.process_multiple_batches(self._sgd_step,
                                               self._sgd_steps_per_learner_step)
    self._sgd_steps = jax.pmap(
        sgd_steps, axis_name=_PMAP_AXIS_NAME, devices=self._devices)

    self._iterator = iterator

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None
    self._current_step = 0

  def _distributional_loss(self, prediction: mpo_types.DistributionLike,
                           target: chex.Array):
    """Compute the critic loss given the prediction and target."""
    # TODO(abef): break this function into separate functions for each critic.
    chex.assert_rank(target, 3)  # [N, Z, T] except for Categorical is [1, T, L]
    if self._critic_type == CriticType.MIXTURE_OF_GAUSSIANS:
      # Sample-based cross-entropy loss.
      loss = -prediction.log_prob(target[..., jnp.newaxis])
      loss = jnp.mean(loss, axis=[0, 1])  # [T]
    elif self._critic_type == CriticType.NONDISTRIBUTIONAL:
      # TD error.
      prediction = prediction.squeeze(axis=-1)  # [T]
      loss = 0.5 * jnp.square(target - prediction)
      chex.assert_equal_shape([target, loss])  # Check broadcasting.
    elif self._critic_type == mpo_types.CriticType.CATEGORICAL_2HOT:
      # Cross-entropy loss (two-hot categorical).
      target = jnp.mean(target, axis=(0, 1))  # [N, Z, T] -> [T]
      # TODO(abef): Compute target differently? (e.g., do mean cross ent.).
      target_probs = rlax.transform_to_2hot(  # [T, L]
          target,
          min_value=prediction.values.min(),
          max_value=prediction.values.max(),
          num_bins=prediction.logits.shape[-1])
      logits = jnp.squeeze(prediction.logits, axis=1)  # [T, L]
      chex.assert_equal_shape([target_probs, logits])
      loss = jax.vmap(rlax.categorical_cross_entropy)(target_probs, logits)
    elif self._critic_type == mpo_types.CriticType.CATEGORICAL:
      loss = jax.vmap(rlax.categorical_cross_entropy)(jnp.squeeze(
          target, axis=0), jnp.squeeze(prediction.logits, axis=1))
    return jnp.mean(loss)  # [T] -> []

  def _compute_predictions(self, params: mpo_networks.MPONetworkParams,
                           sequence: adders.Step) -> mpo_types.ModelOutputs:
    """Compute model predictions at observed and rolled out states."""

    # Initialize the core states, possibly to the recorded stale state.
    if self._use_stale_state:
      initial_state = utils.maybe_recover_lstm_type(
          sequence.extras['core_state'])
      initial_state = tree.map_structure(lambda x: x[0], initial_state)
    else:
      initial_state = self._networks.torso.initial_state_fn(
          params.torso_initial_state, None)

    # Unroll the online core network. Note that this may pass the embeddings
    # unchanged if, say, the core is an hk.IdentityCore.
    state_embedding, _ = self._networks.torso_unroll(   # [T, ...]
        params, sequence.observation, initial_state)

    # Compute the root policy and critic outputs; [T, ...] and [T-1, ...].
    policy = self._networks.policy_head_apply(params, state_embedding)
    q_value = self._networks.critic_head_apply(
        params, state_embedding[:-1], sequence.action[:-1])

    return mpo_types.ModelOutputs(
        policy=policy,  # [T, ...]
        q_value=q_value,  # [T-1, ...]
        reward=None,
        embedding=state_embedding)  # [T, ...]

  def _compute_targets(
      self,
      target_params: mpo_networks.MPONetworkParams,
      dual_params: mpo_types.DualParams,
      sequence: adders.Step,
      online_policy: types.NestedArray,  # TODO(abef): remove this.
      key: jax_types.PRNGKey) -> mpo_types.LossTargets:
    """Compute the targets needed to train the agent."""

    # Initialize the core states, possibly to the recorded stale state.
    if self._use_stale_state:
      initial_state = utils.maybe_recover_lstm_type(
          sequence.extras['core_state'])
      initial_state = tree.map_structure(lambda x: x[0], initial_state)
    else:
      initial_state = self._networks.torso.initial_state_fn(
          target_params.torso_initial_state, None)

    # Unroll the target core network. Note that this may pass the embeddings
    # unchanged if, say, the core is an hk.IdentityCore.
    target_state_embedding, _ = self._networks.torso_unroll(
        target_params, sequence.observation, initial_state)  # [T, ...]

    # Compute the action distribution from target policy network.
    target_policy = self._networks.policy_head_apply(
        target_params, target_state_embedding)  # [T, ...]

    # Maybe reward clip.
    clipped_reward = jnp.clip(sequence.reward, *self._reward_clip_range)  # [T]
    # TODO(abef): when to clip rewards, if at all, if learning dynamics model?

    @jax.named_call
    @jax.vmap
    def critic_mean_fn(action_: jnp.ndarray) -> jnp.ndarray:
      """Compute mean of target critic distribution."""
      critic_output = self._networks.critic_head_apply(
          target_params, target_state_embedding, action_)
      if self._critic_type != CriticType.NONDISTRIBUTIONAL:
        critic_output = critic_output.mean()
      return critic_output

    @jax.named_call
    @jax.vmap
    def critic_sample_fn(action_: jnp.ndarray,
                         seed_: jnp.ndarray) -> jnp.ndarray:
      """Sample from the target critic distribution."""
      z_distribution = self._networks.critic_head_apply(
          target_params, target_state_embedding, action_)
      z_samples = z_distribution.sample(
          self._policy_eval_num_val_samples, seed=seed_)
      return z_samples  # [Z, T, 1]

    if self._discrete_policy:
      # Use all actions to improve policy (no sampling); N = num_actions.
      a_improvement = jnp.arange(self._action_spec.num_values)  # [N]
      seq_len = target_state_embedding.shape[0]  # T
      a_improvement = jnp.tile(a_improvement[..., None], [1, seq_len])  # [N, T]
    else:
      # Sample actions to improve policy; [N=num_samples, T].
      a_improvement = target_policy.sample(self._num_samples, seed=key)

    # TODO(abef): use model to get q_improvement = r + gamma*V?

    # Compute the mean Q-values used in policy improvement; [N, T].
    q_improvement = critic_mean_fn(a_improvement).squeeze(axis=-1)

    # Policy to use for policy evaluation and bootstrapping.
    if self._use_online_policy_to_bootstrap:
      policy_to_evaluate = online_policy
      chex.assert_equal(online_policy.batch_shape, target_policy.batch_shape)
    else:
      policy_to_evaluate = target_policy

    # Action(s) to use for policy evaluation; shape [N, T].
    if self._policy_eval_stochastic:
      a_evaluation = policy_to_evaluate.sample(self._num_samples, seed=key)
    else:
      a_evaluation = policy_to_evaluate.mode()
      a_evaluation = jnp.expand_dims(a_evaluation, axis=0)  # [N=1, T]

    # TODO(abef): policy_eval_stochastic=False makes our targets more "greedy"

    # Add a stopgrad in case we use the online policy for evaluation.
    a_evaluation = jax.lax.stop_gradient(a_evaluation)

    if self._critic_type == CriticType.MIXTURE_OF_GAUSSIANS:
      # Produce Z return samples for every N action sample; [N, Z, T, 1].
      seeds = jax.random.split(key, num=a_evaluation.shape[0])
      z_samples = critic_sample_fn(a_evaluation, seeds)
    else:
      normalized_weights = 1. / a_evaluation.shape[0]
      z_samples = critic_mean_fn(a_evaluation)  # [N, T, 1]

      # When policy_eval_stochastic == True, this corresponds to expected SARSA.
      # Otherwise, normalized_weights = 1.0 and N = 1 so the sum is a no-op.
      z_samples = jnp.sum(normalized_weights * z_samples, axis=0, keepdims=True)
      z_samples = jnp.expand_dims(z_samples, axis=1)  # [N, Z=1, T, 1]

    # Slice to t = 1...T and transform into raw reward space; [N, Z, T].
    z_samples_itx = self._tx_pair.apply_inv(z_samples.squeeze(axis=-1))

    # Compute the value estimate by averaging the sampled returns in the raw
    # reward space; shape [N=1, Z=1, T].
    value_target_itx = jnp.mean(z_samples_itx, axis=(0, 1), keepdims=True)

    if self._use_retrace:
      # Warning! Retrace has not been tested with the MoG critic.
      log_rhos = (
          target_policy.log_prob(sequence.action) - sequence.extras['log_prob'])

      # Compute Q-values; expand and squeeze because critic_mean_fn is vmapped.
      q_t = critic_mean_fn(jnp.expand_dims(sequence.action, axis=0)).squeeze(0)
      q_t = q_t.squeeze(-1)  # Also squeeze trailing scalar dimension; [T].

      # Compute retrace targets.
      # These targets use the rewards and discounts as in normal TD-learning but
      # they use a mix of bootstrapped values V(s') and Q(s', a'), weighing the
      # latter based on how likely a' is under the current policy (s' and a' are
      # samples from replay).
      # See [Munos et al., 2016](https://arxiv.org/abs/1606.02647) for more.
      q_value_target_itx = rlax.general_off_policy_returns_from_q_and_v(
          q_t=self._tx_pair.apply_inv(q_t[1:-1]),
          v_t=jnp.squeeze(value_target_itx, axis=(0, 1))[1:],
          r_t=clipped_reward[:-1],
          discount_t=self._discount * sequence.discount[:-1],
          c_t=self._retrace_lambda * jnp.minimum(1.0, jnp.exp(log_rhos[1:-1])))

      # Expand dims to the expected [N=1, Z=1, T-1].
      q_value_target_itx = jnp.expand_dims(q_value_target_itx, axis=(0, 1))
    else:
      # Compute bootstrap target from sequences. vmap return computation across
      # N action and Z return samples; shape [N, Z, T-1].
      n_step_return_fn = functools.partial(
          rlax.n_step_bootstrapped_returns,
          r_t=clipped_reward[:-1],
          discount_t=self._discount * sequence.discount[:-1],
          n=self._n_step_for_sequence_bootstrap,
          lambda_t=self._td_lambda)
      n_step_return_vfn = jax.vmap(jax.vmap(n_step_return_fn))
      q_value_target_itx = n_step_return_vfn(v_t=z_samples_itx[..., 1:])

    # Transform back to the canonical space and stop gradients.
    q_value_target = jax.lax.stop_gradient(
        self._tx_pair.apply(q_value_target_itx))
    reward_target = jax.lax.stop_gradient(self._tx_pair.apply(clipped_reward))
    value_target = jax.lax.stop_gradient(self._tx_pair.apply(value_target_itx))

    if self._critic_type == mpo_types.CriticType.CATEGORICAL:

      @jax.vmap
      def get_logits_and_values(
          action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic_output = self._networks.critic_head_apply(
            target_params, target_state_embedding[1:], action)
        return critic_output.logits, critic_output.values

      z_t_logits, z_t_values = get_logits_and_values(a_evaluation[:, 1:])
      z_t_logits = jnp.squeeze(z_t_logits, axis=2)  # [N, T-1, L]
      z_t_values = z_t_values[0]  # Values are identical at each N; [L].

      gamma = self._discount * sequence.discount[:-1, None]  # [T-1, 1]
      r_t = clipped_reward[:-1, None]  # [T-1, 1]
      atoms_itx = self._tx_pair.apply_inv(z_t_values)[None, ...]  # [1, L]
      z_target_atoms = self._tx_pair.apply(r_t + gamma * atoms_itx)  # [T-1, L]
      # Note: this is n=1-step TD unless using experience=FromTransitions(n>1).
      z_target_probs = jax.nn.softmax(z_t_logits)  # [N, T-1, L]
      z_target_atoms = jax.lax.broadcast(
          z_target_atoms, z_target_probs.shape[:1])  # [N, T-1, L]
      project_fn = functools.partial(
          rlax.categorical_l2_project, z_q=z_t_values)
      z_target = jax.vmap(jax.vmap(project_fn))(z_target_atoms, z_target_probs)
      z_target = jnp.mean(z_target, axis=0)
      q_value_target = jax.lax.stop_gradient(z_target[None, ...])  # [1, T-1, L]
      # TODO(abef): make q_v_target shape align with expected [N, Z, T-1] shape?

    targets = mpo_types.LossTargets(
        policy=target_policy,  # [T, ...]
        a_improvement=a_improvement,  # [N, T]
        q_improvement=q_improvement,  # [N, T]
        q_value=q_value_target,  # [N, Z, T-1] ([1, T-1, L] for CATEGORICAL)
        value=value_target[..., :-1],  # [N=1, Z=1, T-1]
        reward=reward_target,  # [T]
        embedding=target_state_embedding)  # [T, ...]

    return targets

  def _loss_fn(
      self,
      params: mpo_networks.MPONetworkParams,
      dual_params: mpo_types.DualParams,
      # TODO(bshahr): clean up types: Step is not a great type for sequences.
      sequence: adders.Step,
      target_params: mpo_networks.MPONetworkParams,
      key: jax_types.PRNGKey) -> Tuple[jnp.ndarray, mpo_types.LogDict]:
    # Compute the model predictions at the root and for the rollouts.
    predictions = self._compute_predictions(params=params, sequence=sequence)

    # Compute the targets to use for the losses.
    targets = self._compute_targets(
        target_params=target_params,
        dual_params=dual_params,
        sequence=sequence,
        online_policy=predictions.policy,
        key=key)

    # TODO(abef): mask policy loss at terminal states or use uniform targets
    # is_terminal = sequence.discount == 0.

    # Compute MPO policy loss on each state in the sequence.
    policy_loss, policy_stats = self._policy_loss_module(
        params=dual_params,
        online_action_distribution=predictions.policy,  # [T, ...].
        target_action_distribution=targets.policy,  # [T, ...].
        actions=targets.a_improvement,  # Unused in discrete case; [N, T].
        q_values=targets.q_improvement)  # [N, T]

    # Compute the critic loss on the states in the sequence.
    critic_loss = self._distributional_loss(
        prediction=predictions.q_value,  # [T-1, 1, ...]
        target=targets.q_value)  # [N, Z, T-1]

    loss = (self._loss_scales.policy * policy_loss +
            self._loss_scales.critic * critic_loss)
    loss_logging_dict = {
        'loss': loss,
        'root_policy_loss': policy_loss,
        'root_critic_loss': critic_loss,
        'policy_loss': policy_loss,
        'critic_loss': critic_loss,
    }

    # Append MPO statistics.
    loss_logging_dict.update(
        {f'policy/root/{k}': v for k, v in policy_stats._asdict().items()})

    # Compute rollout losses.
    if self._model_rollout_length > 0:
      model_rollout_loss, rollout_logs = self._model_rollout_loss_fn(
          params, dual_params, sequence, predictions.embedding, targets, key)
      loss += model_rollout_loss
      loss_logging_dict.update(rollout_logs)
      loss_logging_dict.update({
          'policy_loss': policy_loss + rollout_logs['rollout_policy_loss'],
          'critic_loss': critic_loss + rollout_logs['rollout_critic_loss'],
          'loss': loss})

    return loss, loss_logging_dict

  def _sgd_step(
      self,
      state: TrainingState,
      transitions: Union[types.Transition, adders.Step],
  ) -> Tuple[TrainingState, Dict[str, Any]]:
    """Perform one parameter update step."""

    if isinstance(transitions, types.Transition):
      sequences = mpo_utils.make_sequences_from_transitions(transitions)
      if self._model_rollout_length > 0:
        raise ValueError('model rollouts not yet supported from transitions')
    else:
      sequences = transitions

    # Get next random_key and `batch_size` keys.
    batch_size = sequences.reward.shape[0]
    keys = jax.random.split(state.random_key, num=batch_size+1)
    random_key, keys = keys[0], keys[1:]

    # Vmap over the batch dimension when learning from sequences.
    loss_vfn = jax.vmap(self._loss_fn, in_axes=(None, None, 0, None, 0))
    safe_mean = lambda x: jnp.mean(x) if x is not None else x
    # TODO(bshahr): Consider cleaning this up via acme.tree_utils.tree_map.
    loss_fn = lambda *a, **k: tree.map_structure(safe_mean, loss_vfn(*a, **k))

    loss_and_grad = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)

    # Compute the loss and gradient.
    (_, loss_log_dict), all_gradients = loss_and_grad(
        state.params, state.dual_params, sequences, state.target_params, keys)

    # Average gradients across replicas.
    gradients, dual_gradients = jax.lax.pmean(all_gradients, _PMAP_AXIS_NAME)

    # Compute gradient norms before clipping.
    gradients_norm = optax.global_norm(gradients)
    dual_gradients_norm = optax.global_norm(dual_gradients)

    # Get optimizer updates and state.
    updates, opt_state = self._optimizer.update(
        gradients, state.opt_state, state.params)
    dual_updates, dual_opt_state = self._dual_optimizer.update(
        dual_gradients, state.dual_opt_state, state.dual_params)

    # Apply optimizer updates to parameters.
    params = optax.apply_updates(state.params, updates)
    dual_params = optax.apply_updates(state.dual_params, dual_updates)

    # Clip dual params at some minimum value.
    dual_params = self._dual_clip_fn(dual_params)

    steps = state.steps + 1

    # Periodically update target networks.
    if self._target_update_period:
      target_params = optax.periodic_update(params, state.target_params, steps,  # pytype: disable=wrong-arg-types  # numpy-scalars
                                            self._target_update_period)
    elif self._target_update_rate:
      target_params = optax.incremental_update(params, state.target_params,
                                               self._target_update_rate)

    new_state = TrainingState(  # pytype: disable=wrong-arg-types  # numpy-scalars
        params=params,
        target_params=target_params,
        dual_params=dual_params,
        opt_state=opt_state,
        dual_opt_state=dual_opt_state,
        steps=steps,
        random_key=random_key,
    )

    # Log the metrics from this learner step.
    metrics = {f'loss/{k}': v for k, v in loss_log_dict.items()}

    metrics.update({
        'opt/grad_norm': gradients_norm,
        'opt/param_norm': optax.global_norm(params)})

    dual_metrics = {
        'opt/dual_grad_norm': dual_gradients_norm,
        'opt/dual_param_norm': optax.global_norm(dual_params),
        'params/dual/log_temperature_avg': dual_params.log_temperature}
    if isinstance(dual_params, continuous_losses.MPOParams):
      dual_metrics.update({
          'params/dual/log_alpha_mean_avg': dual_params.log_alpha_mean,
          'params/dual/log_alpha_stddev_avg': dual_params.log_alpha_stddev})
      if dual_params.log_penalty_temperature is not None:
        dual_metrics['params/dual/log_penalty_temp_mean'] = (
            dual_params.log_penalty_temperature)
    elif isinstance(dual_params, discrete_losses.CategoricalMPOParams):
      dual_metrics['params/dual/log_alpha_avg'] = dual_params.log_alpha
    metrics.update(jax.tree.map(jnp.mean, dual_metrics))

    return new_state, metrics

  def step(self):
    """Perform one learner step, which in general does multiple SGD steps."""
    with jax.profiler.StepTraceAnnotation('step', step_num=self._current_step):
      # Get data from replay (dropping extras if any). Note there is no
      # extra data here because we do not insert any into Reverb.
      sample = next(self._iterator)
      if isinstance(self._experience_type, mpo_types.FromTransitions):
        minibatch = types.Transition(*sample.data)
      elif isinstance(self._experience_type, mpo_types.FromSequences):
        minibatch = adders.Step(*sample.data)

      self._state, metrics = self._sgd_steps(self._state, minibatch)
      self._current_step, metrics = mpo_utils.get_from_first_device(
          (self._state.steps, metrics))

      # Compute elapsed time.
      timestamp = time.time()
      elapsed_time = timestamp - self._timestamp if self._timestamp else 0
      self._timestamp = timestamp

      # Increment counts and record the current time
      counts = self._counter.increment(
          steps=self._sgd_steps_per_learner_step, walltime=elapsed_time)

      if elapsed_time > 0:
        metrics['steps_per_second'] = (
            self._sgd_steps_per_learner_step / elapsed_time)
      else:
        metrics['steps_per_second'] = 0.

      # Attempts to write the logs.
      if self._logger:
        self._logger.write({**metrics, **counts})

  def get_variables(self, names: List[str]) -> network_lib.Params:
    params = mpo_utils.get_from_first_device(self._state.target_params)

    variables = {
        'policy_head': params.policy_head,
        'critic_head': params.critic_head,
        'torso': params.torso,
        'network': params,
        'policy': params._replace(critic_head={}),
        'critic': params._replace(policy_head={}),
    }
    return [variables[name] for name in names]

  def save(self) -> TrainingState:
    return jax.tree.map(mpo_utils.get_from_first_device, self._state)

  def restore(self, state: TrainingState):
    self._state = utils.replicate_in_all_devices(state, self._local_devices)


def _get_default_optimizer(
    learning_rate: optax.ScalarOrSchedule, max_grad_norm: Optional[float] = None
) -> optax.GradientTransformation:
  optimizer = optax.adam(learning_rate)
  if max_grad_norm and max_grad_norm > 0:
    optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), optimizer)
  return optimizer
