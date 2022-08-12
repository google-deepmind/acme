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

"""R2D2 Builder."""
from typing import Generic, Iterator, List, Optional

import acme
from acme import adders
from acme import core
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.adders.reverb import base as reverb_base
from acme.adders.reverb import structured
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.agents.jax.r2d2 import actor as r2d2_actor
from acme.agents.jax.r2d2 import config as r2d2_config
from acme.agents.jax.r2d2 import learning as r2d2_learning
from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.datasets import reverb as datasets
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import jax
import optax
import reverb
from reverb import structured_writer as sw
import tensorflow as tf
import tree

# TODO(b/450949030): extrac the private functions to a library once other agents
# reuse them.

# TODO(b/450949030): add support to add all the final subsequences of
# length < sequence_lenght at the end of the episode and pad them with zeros.
# We have to check if this requires moving _zero_pad to the adder.


def _build_sequence(length: int,
                    step_spec: reverb_base.Step) -> reverb_base.Trajectory:
  """Constructs the sequence using only the first value of core_state."""
  step_dict = step_spec._asdict()
  extras_dict = step_dict.pop('extras')
  return reverb_base.Trajectory(
      **tree.map_structure(lambda x: x[-length:], step_dict),
      extras=tree.map_structure(lambda x: x[-length], extras_dict))


def _zero_pad(sequence_length: int) -> datasets.Transform:
  """Adds zero padding to the right so all samples have the same length."""

  def _zero_pad_transform(sample: reverb.ReplaySample) -> reverb.ReplaySample:
    trajectory: reverb_base.Trajectory = sample.data

    # Split steps and extras data (the extras won't be padded as they only
    # contain one element)
    trajectory_steps = trajectory._asdict()
    trajectory_extras = trajectory_steps.pop('extras')

    unpadded_length = len(tree.flatten(trajectory_steps)[0])

    # Do nothing if the sequence is already full.
    if unpadded_length != sequence_length:
      to_pad = sequence_length - unpadded_length
      pad = lambda x: tf.pad(x, [[0, to_pad]] + [[0, 0]] * (len(x.shape) - 1))

      trajectory_steps = tree.map_structure(pad, trajectory_steps)

    # Set the shape to be statically known, and checks it at runtime.
    def _ensure_shape(x):
      shape = tf.TensorShape([sequence_length]).concatenate(x.shape[1:])
      return tf.ensure_shape(x, shape)

    trajectory_steps = tree.map_structure(_ensure_shape, trajectory_steps)
    return reverb.ReplaySample(
        info=sample.info,
        data=reverb_base.Trajectory(
            **trajectory_steps, extras=trajectory_extras))

  return _zero_pad_transform


def _make_adder_config(step_spec: reverb_base.Step, seq_len: int,
                       seq_period: int) -> list[sw.Config]:
  return structured.create_sequence_config(
      step_spec=step_spec,
      sequence_length=seq_len,
      period=seq_period,
      end_of_episode_behavior=adders_reverb.EndBehavior.TRUNCATE,
      sequence_pattern=_build_sequence)


class R2D2Builder(Generic[actor_core_lib.RecurrentState],
                  builders.ActorLearnerBuilder[r2d2_networks.R2D2Networks,
                                               r2d2_actor.R2D2Policy,
                                               r2d2_learning.R2D2ReplaySample]):
  """R2D2 Builder.

  This is constructs all of the components for Recurrent Experience Replay in
  Distributed Reinforcement Learning (Kapturowski et al.)
  https://openreview.net/pdf?id=r1lyTjAqYX.
  """

  def __init__(self, config: r2d2_config.R2D2Config):
    """Creates a R2D2 learner, a behavior policy and an eval actor."""
    self._config = config
    self._sequence_length = (
        self._config.burn_in_length + self._config.trace_length + 1)

  @property
  def _batch_size_per_device(self) -> int:
    """Splits batch size across all learner devices evenly."""
    # TODO(bshahr): Using jax.device_count will not be valid when colocating
    # learning and inference.
    return self._config.batch_size // jax.device_count()

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: r2d2_networks.R2D2Networks,
      dataset: Iterator[r2d2_learning.R2D2ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    del environment_spec

    # The learner updates the parameters (and initializes them).
    return r2d2_learning.R2D2Learner(
        unroll=networks.unroll,
        initial_state=networks.initial_state,
        batch_size=self._batch_size_per_device,
        random_key=random_key,
        burn_in_length=self._config.burn_in_length,
        discount=self._config.discount,
        importance_sampling_exponent=(
            self._config.importance_sampling_exponent),
        max_priority_weight=self._config.max_priority_weight,
        target_update_period=self._config.target_update_period,
        iterator=dataset,
        optimizer=optax.adam(self._config.learning_rate),
        bootstrap_n=self._config.bootstrap_n,
        tx_pair=self._config.tx_pair,
        clip_rewards=self._config.clip_rewards,
        replay_client=replay_client,
        counter=counter,
        logger=logger_fn('learner'))

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy: r2d2_actor.R2D2Policy,
  ) -> List[reverb.Table]:
    """Create tables to insert data into."""
    dummy_actor_state = policy.init(jax.random.PRNGKey(0))
    extras_spec = policy.get_extras(dummy_actor_state)
    step_spec = structured.create_step_spec(
        environment_spec=environment_spec, extras_spec=extras_spec)
    if self._config.samples_per_insert:
      samples_per_insert_tolerance = (
          self._config.samples_per_insert_tolerance_rate *
          self._config.samples_per_insert)
      error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
      limiter = reverb.rate_limiters.SampleToInsertRatio(
          min_size_to_sample=self._config.min_replay_size,
          samples_per_insert=self._config.samples_per_insert,
          error_buffer=error_buffer)
    else:
      limiter = reverb.rate_limiters.MinSize(self._config.min_replay_size)
    return [
        reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Prioritized(
                self._config.priority_exponent),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=limiter,
            signature=sw.infer_signature(
                configs=_make_adder_config(step_spec, self._sequence_length,
                                           self._config.sequence_period),
                step_spec=step_spec))
    ]

  def make_dataset_iterator(
      self,
      replay_client: reverb.Client) -> Iterator[r2d2_learning.R2D2ReplaySample]:
    """Create a dataset iterator to use for learning/updating the agent."""
    batch_size_per_learner = self._config.batch_size // jax.process_count()
    dataset = datasets.make_reverb_dataset(
        table=self._config.replay_table_name,
        server_address=replay_client.server_address,
        batch_size=self._batch_size_per_device,
        num_parallel_calls=None,
        max_in_flight_samples_per_worker=2 * batch_size_per_learner,
        postprocess=_zero_pad(self._sequence_length),
    )

    # We split samples in two outputs, the keys which need to be kept on-host
    # since int64 arrays are not supported in TPUs, and the entire sample
    # separately so it can be sent to the sgd_step method.
    def split_sample(sample: reverb.ReplaySample) -> utils.PrefetchingSplit:
      return utils.PrefetchingSplit(host=sample.info.key, device=sample)

    return utils.multi_device_put(
        dataset.as_numpy_iterator(),
        devices=jax.local_devices(),
        split_fn=split_sample)

  def make_adder(
      self, replay_client: reverb.Client,
      environment_spec: Optional[specs.EnvironmentSpec],
      policy: Optional[r2d2_actor.R2D2Policy]) -> Optional[adders.Adder]:
    """Create an adder which records data generated by the actor/environment."""
    if environment_spec is None or policy is None:
      raise ValueError('`environment_spec` and `policy` cannot be None.')
    dummy_actor_state = policy.init(jax.random.PRNGKey(0))
    extras_spec = policy.get_extras(dummy_actor_state)
    step_spec = structured.create_step_spec(
        environment_spec=environment_spec, extras_spec=extras_spec)
    return structured.StructuredAdder(
        client=replay_client,
        max_in_flight_items=5,
        configs=_make_adder_config(step_spec, self._sequence_length,
                                   self._config.sequence_period),
        step_spec=step_spec)

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy: r2d2_actor.R2D2Policy,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
      adder: Optional[adders.Adder] = None,
  ) -> acme.Actor:
    del environment_spec
    # Create variable client.
    variable_client = variable_utils.VariableClient(
        variable_source,
        key='actor_variables',
        update_period=self._config.variable_update_period)

    return actors.GenericActor(
        policy, random_key, variable_client, adder, backend='cpu')

  def make_policy(self,
                  networks: r2d2_networks.R2D2Networks,
                  environment_spec: specs.EnvironmentSpec,
                  evaluation: bool = False) -> r2d2_actor.R2D2Policy:
    if evaluation:
      return r2d2_actor.get_actor_core(
          networks,
          num_epsilons=None,
          evaluation_epsilon=self._config.evaluation_epsilon)
    else:
      return r2d2_actor.get_actor_core(networks, self._config.num_epsilons)
