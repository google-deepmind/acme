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

"""Defines the MPO agent builder, which holds factories for all components."""

import functools
from typing import Iterator, List, Optional

from absl import logging
from acme import core
from acme import specs
from acme.adders import base
from acme.adders import reverb as adders
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.agents.jax.mpo import acting
from acme.agents.jax.mpo import config as mpo_config
from acme.agents.jax.mpo import learning
from acme.agents.jax.mpo import networks as mpo_networks
from acme.agents.jax.mpo import types as mpo_types
from acme.datasets import image_augmentation as img_aug
from acme.datasets import reverb as datasets
from acme.jax import observation_stacking as obs_stacking
from acme.jax import types as jax_types
from acme.jax import utils
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import jax
import optax
import reverb
# Acme loves Reverb.
import tensorflow as tf
import tree

_POLICY_KEY = 'policy'
_QUEUE_TABLE_NAME = 'queue_table'


class MPOBuilder(builders.ActorLearnerBuilder):
  """Builder class for MPO agent components."""

  def __init__(self,
               config: mpo_config.MPOConfig,
               *,
               sgd_steps_per_learner_step: int = 8,
               max_learner_steps: Optional[int] = None):
    self.config = config
    self.sgd_steps_per_learner_step = sgd_steps_per_learner_step
    self._max_learner_steps = max_learner_steps

  def make_policy(
      self,
      networks: mpo_networks.MPONetworks,
      environment_spec: specs.EnvironmentSpec,
      evaluation: bool = False,
  ) -> actor_core_lib.ActorCore:
    actor_core = acting.make_actor_core(
        networks,
        stochastic=not evaluation,
        store_core_state=self.config.use_stale_state,
        store_log_prob=self.config.use_retrace)

    # Maybe wrap the actor core to perform actor-side observation stacking.
    if self.config.num_stacked_observations > 1:
      actor_core = obs_stacking.wrap_actor_core(
          actor_core,
          observation_spec=environment_spec.observations,
          num_stacked_observations=self.config.num_stacked_observations)

    return actor_core

  def make_actor(
      self,
      random_key: jax_types.PRNGKey,
      policy: actor_core_lib.ActorCore,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
      adder: Optional[base.Adder] = None,
  ) -> core.Actor:

    del environment_spec  # This actor doesn't need the spec beyond the policy.
    variable_client = variable_utils.VariableClient(
        client=variable_source,
        key=_POLICY_KEY,
        update_period=self.config.variable_update_period)

    return actors.GenericActor(
        actor=policy,
        random_key=random_key,
        variable_client=variable_client,
        adder=adder,
        backend='cpu')

  def make_learner(self,
                   random_key: jax_types.PRNGKey,
                   networks: mpo_networks.MPONetworks,
                   dataset: Iterator[reverb.ReplaySample],
                   logger_fn: loggers.LoggerFactory,
                   environment_spec: specs.EnvironmentSpec,
                   replay_client: Optional[reverb.Client] = None,
                   counter: Optional[counting.Counter] = None) -> core.Learner:
    # Set defaults.
    del replay_client  # Unused as we do not update priorities.
    learning_rate = self.config.learning_rate

    # Make sure we can split the batches evenly across all accelerator devices.
    num_learner_devices = jax.device_count()
    if self.config.batch_size % num_learner_devices > 0:
      raise ValueError(
          'Batch size must divide evenly by the number of learner devices.'
          f' Passed a batch size of {self.config.batch_size} and the number of'
          f' available learner devices is {num_learner_devices}. Specifically,'
          f' devices: {jax.devices()}.')

    agent_environment_spec = environment_spec
    if self.config.num_stacked_observations > 1:
      # Adjust the observation spec for the agent-side frame-stacking.
      # Note: this is only for the ActorCore's benefit, the adders want the true
      # environment spec.
      agent_environment_spec = obs_stacking.get_adjusted_environment_spec(
          agent_environment_spec, self.config.num_stacked_observations)

    if self.config.use_cosine_lr_decay:
      learning_rate = optax.warmup_cosine_decay_schedule(
          init_value=0.,
          peak_value=self.config.learning_rate,
          warmup_steps=self.config.cosine_lr_decay_warmup_steps,
          decay_steps=self._max_learner_steps)

    optimizer = optax.adamw(
        learning_rate,
        b1=self.config.adam_b1,
        b2=self.config.adam_b2,
        weight_decay=self.config.weight_decay)
    # TODO(abef): move LR scheduling and optimizer creation into launcher.

    loss_scales_config = mpo_types.LossScalesConfig(
        policy=self.config.policy_loss_scale,
        critic=self.config.critic_loss_scale,
        rollout=mpo_types.RolloutLossScalesConfig(
            policy=self.config.rollout_policy_loss_scale,
            bc_policy=self.config.rollout_bc_policy_loss_scale,
            critic=self.config.rollout_critic_loss_scale,
            reward=self.config.rollout_reward_loss_scale,
        ))

    logger = logger_fn(
        'learner',
        steps_key=counter.get_steps_key() if counter else 'learner_steps')

    with jax.disable_jit(not self.config.jit_learner):
      learner = learning.MPOLearner(
          iterator=dataset,
          networks=networks,
          environment_spec=agent_environment_spec,
          critic_type=self.config.critic_type,
          discrete_policy=self.config.discrete_policy,
          random_key=random_key,
          discount=self.config.discount,
          num_samples=self.config.num_samples,
          policy_eval_stochastic=self.config.policy_eval_stochastic,
          policy_eval_num_val_samples=self.config.policy_eval_num_val_samples,
          policy_loss_config=self.config.policy_loss_config,
          loss_scales=loss_scales_config,
          target_update_period=self.config.target_update_period,
          target_update_rate=self.config.target_update_rate,
          experience_type=self.config.experience_type,
          use_online_policy_to_bootstrap=(
              self.config.use_online_policy_to_bootstrap),
          use_stale_state=self.config.use_stale_state,
          use_retrace=self.config.use_retrace,
          retrace_lambda=self.config.retrace_lambda,
          model_rollout_length=self.config.model_rollout_length,
          sgd_steps_per_learner_step=self.sgd_steps_per_learner_step,
          optimizer=optimizer,
          dual_optimizer=optax.adam(self.config.dual_learning_rate),
          grad_norm_clip=self.config.grad_norm_clip,
          reward_clip=self.config.reward_clip,
          value_tx_pair=self.config.value_tx_pair,
          counter=counter,
          logger=logger,
          devices=jax.devices(),
      )
    return learner

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy: actor_core_lib.ActorCore,  # Used to get accurate extras_spec.
  ) -> List[reverb.Table]:
    dummy_actor_state = policy.init(jax.random.PRNGKey(0))
    extras_spec = policy.get_extras(dummy_actor_state)

    if isinstance(self.config.experience_type, mpo_types.FromTransitions):
      signature = adders.NStepTransitionAdder.signature(environment_spec,
                                                        extras_spec)
    elif isinstance(self.config.experience_type, mpo_types.FromSequences):
      sequence_length = (
          self.config.experience_type.sequence_length +
          self.config.num_stacked_observations - 1)
      signature = adders.SequenceAdder.signature(
          environment_spec, extras_spec, sequence_length=sequence_length)
    # TODO(bshahr): This way of obtaining the signature is error-prone. Find a
    # programmatic way via make_adder.

    # Create the rate limiter.
    if self.config.samples_per_insert:
      # Create enough of an error buffer to give a 10% tolerance in rate.
      samples_per_insert_tolerance = 0.1 * self.config.samples_per_insert
      error_buffer = self.config.min_replay_size * samples_per_insert_tolerance
      limiter = reverb.rate_limiters.SampleToInsertRatio(
          min_size_to_sample=self.config.min_replay_size,
          samples_per_insert=self.config.samples_per_insert,
          error_buffer=max(error_buffer, 2 * self.config.samples_per_insert))
    else:
      limiter = reverb.rate_limiters.MinSize(self.config.min_replay_size)

    # Reverb loves Acme.
    replay_extensions = []
    queue_extensions = []


    # Create replay tables.
    tables = []
    if self.config.replay_fraction > 0:
      replay_table = reverb.Table(
          name=adders.DEFAULT_PRIORITY_TABLE,
          sampler=reverb.selectors.Uniform(),
          remover=reverb.selectors.Fifo(),
          max_size=self.config.max_replay_size,
          rate_limiter=limiter,
          extensions=replay_extensions,
          signature=signature)
      tables.append(replay_table)
      logging.info(
          'Creating off-policy replay buffer with replay fraction %g '
          'of batch %d', self.config.replay_fraction, self.config.batch_size)

    if self.config.replay_fraction < 1:
      # Create a FIFO queue. This will provide the rate limitation if used.
      queue = reverb.Table.queue(
          name=_QUEUE_TABLE_NAME,
          max_size=self.config.online_queue_capacity,
          extensions=queue_extensions,
          signature=signature)
      tables.append(queue)
      logging.info(
          'Creating online replay queue with queue fraction %g '
          'of batch %d', 1.0 - self.config.replay_fraction,
          self.config.batch_size)

    return tables

  def make_adder(
      self,
      replay_client: reverb.Client,
      environment_spec: Optional[specs.EnvironmentSpec],
      policy: Optional[actor_core_lib.ActorCore],
  ) -> Optional[base.Adder]:
    del environment_spec, policy
    # Specify the tables to insert into but don't use prioritization.
    priority_fns = {}
    if self.config.replay_fraction > 0:
      priority_fns[adders.DEFAULT_PRIORITY_TABLE] = None
    if self.config.replay_fraction < 1:
      priority_fns[_QUEUE_TABLE_NAME] = None

    if isinstance(self.config.experience_type, mpo_types.FromTransitions):
      return adders.NStepTransitionAdder(
          client=replay_client,
          n_step=self.config.experience_type.n_step,
          discount=self.config.discount,
          priority_fns=priority_fns)
    elif isinstance(self.config.experience_type, mpo_types.FromSequences):
      sequence_length = (
          self.config.experience_type.sequence_length +
          self.config.num_stacked_observations - 1)
      return adders.SequenceAdder(
          client=replay_client,
          sequence_length=sequence_length,
          period=self.config.experience_type.sequence_period,
          end_of_episode_behavior=adders.EndBehavior.WRITE,
          max_in_flight_items=1,
          priority_fns=priority_fns)

  def make_dataset_iterator(
      self, replay_client: reverb.Client) -> Iterator[reverb.ReplaySample]:

    if self.config.num_stacked_observations > 1:
      maybe_stack_observations = functools.partial(
          obs_stacking.stack_reverb_observation,
          stack_size=self.config.num_stacked_observations)
    else:
      maybe_stack_observations = None

    dataset = datasets.make_reverb_dataset(
        server_address=replay_client.server_address,
        batch_size=self.config.batch_size // jax.device_count(),
        table={
            adders.DEFAULT_PRIORITY_TABLE: self.config.replay_fraction,
            _QUEUE_TABLE_NAME: 1. - self.config.replay_fraction,
        },
        num_parallel_calls=max(16, 4 * jax.local_device_count()),
        max_in_flight_samples_per_worker=(2 * self.sgd_steps_per_learner_step *
                                          self.config.batch_size //
                                          jax.device_count()),
        postprocess=maybe_stack_observations)

    if self.config.observation_transform:
      # Augment dataset with random translations, simulated by pad-and-crop.
      transform = img_aug.make_transform(
          observation_transform=self.config.observation_transform,
          transform_next_observation=isinstance(self.config.experience_type,
                                                mpo_types.FromTransitions))
      dataset = dataset.map(
          transform, num_parallel_calls=16, deterministic=False)

    # Batch and then flatten to feed multiple SGD steps per learner step.
    if self.sgd_steps_per_learner_step > 1:
      dataset = dataset.batch(
          self.sgd_steps_per_learner_step, drop_remainder=True)
      batch_flatten = lambda t: tf.reshape(t, [-1] + t.shape[2:].as_list())
      dataset = dataset.map(lambda x: tree.map_structure(batch_flatten, x))

    return utils.multi_device_put(dataset.as_numpy_iterator(),
                                  jax.local_devices())
