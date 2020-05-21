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

"""Recurrent DQfD (R2D3) agent implementation."""

import functools

from acme import datasets
from acme import specs
from acme import types as acme_types
from acme.adders import reverb as adders
from acme.agents import actors_tf2
from acme.agents import agent
from acme.agents.r2d2 import learning
from acme.utils import counting
from acme.utils import loggers
from acme.utils import tf2_savers
from acme.utils import tf2_utils

import reverb
import sonnet as snt
import tensorflow as tf
import tree
import trfl


class R2D3(agent.Agent):
  """R2D3 Agent.

  This implements a single-process R2D2 agent that mixes demonstrations with
  actor experience.
  """

  def __init__(self,
               environment_spec: specs.EnvironmentSpec,
               network: snt.RNNCore,
               target_network: snt.RNNCore,
               burn_in_length: int,
               trace_length: int,
               replay_period: int,
               demonstration_dataset: tf.data.Dataset,
               demonstration_ratio: float,
               counter: counting.Counter = None,
               logger: loggers.Logger = None,
               discount: float = 0.99,
               batch_size: int = 32,
               target_update_period: int = 100,
               importance_sampling_exponent: float = 0.2,
               epsilon: float = 0.01,
               learning_rate: float = 1e-3,
               log_to_bigtable: bool = False,
               log_name: str = 'agent',
               checkpoint: bool = True,
               min_replay_size: int = 1000,
               max_replay_size: int = 1000000,
               samples_per_insert: float = 32.0):

    replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=max_replay_size,
        rate_limiter=reverb.rate_limiters.MinSize(min_size_to_sample=1))
    self._server = reverb.Server([replay_table], port=None)
    address = f'localhost:{self._server.port}'

    sequence_length = burn_in_length + trace_length + 1
    # Component to add things into replay.
    sequence_kwargs = dict(
        period=replay_period,
        sequence_length=sequence_length,
    )
    adder = adders.SequenceAdder(client=reverb.Client(address),
                                   **sequence_kwargs)

    # The dataset object to learn from.
    reverb_client = reverb.TFClient(address)
    extra_spec = {
        'core_state': network.initial_state(1),
    }
    # Remove batch dimensions.
    extra_spec = tf2_utils.squeeze_batch_dim(extra_spec)
    dataset = datasets.make_reverb_dataset(
        client=reverb_client,
        environment_spec=environment_spec,
        extra_spec=extra_spec,
        sequence_length=sequence_length)

    # Combine with demonstration dataset.
    transition = functools.partial(_sequence_from_episode,
                                   extra_spec=extra_spec,
                                   **sequence_kwargs)
    dataset_demos = demonstration_dataset.map(transition)
    dataset = tf.data.experimental.sample_from_datasets(
        [dataset, dataset_demos],
        [1 - demonstration_ratio, demonstration_ratio])

    # Batch and prefetch.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    tf2_utils.create_variables(network, [environment_spec.observations])
    tf2_utils.create_variables(target_network, [environment_spec.observations])

    learner = learning.R2D2Learner(
        environment_spec=environment_spec,
        network=network,
        target_network=target_network,
        burn_in_length=burn_in_length,
        dataset=dataset,
        reverb_client=reverb_client,
        counter=counter,
        logger=logger,
        sequence_length=sequence_length,
        discount=discount,
        target_update_period=target_update_period,
        importance_sampling_exponent=importance_sampling_exponent,
        max_replay_size=max_replay_size,
        learning_rate=learning_rate,
        store_lstm_state=False,
    )

    self._checkpointer = tf2_savers.Checkpointer(
        subdirectory='r2d2_learner',
        time_delta_minutes=60,
        objects_to_save=learner.state,
        enable_checkpointing=checkpoint,
    )

    self._snapshotter = tf2_savers.Snapshotter(
        objects_to_save={'network': network}, time_delta_minutes=60.)

    policy_network = snt.DeepRNN([
        network,
        lambda qs: trfl.epsilon_greedy(qs, epsilon=epsilon).sample(),
    ])

    actor = actors_tf2.RecurrentActor(policy_network, adder)
    observations_per_step = (float(replay_period * batch_size) /
                             samples_per_insert)
    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=replay_period * max(batch_size, min_replay_size),
        observations_per_step=observations_per_step)

  def update(self):
    super().update()
    self._snapshotter.save()
    self._checkpointer.save()


def _sequence_from_episode(observations: acme_types.NestedTensor,
                           actions: tf.Tensor,
                           rewards: tf.Tensor,
                           discounts: tf.Tensor,
                           extra_spec: acme_types.NestedSpec,
                           period: int,
                           sequence_length: int):
  """Produce Reverb-like sequence from a full episode.

  Observations, actions, rewards and discounts have the same length. This
  function will ignore the first reward and discount and the last action.

  This function generates fake (all-zero) extras.

  See docs for reverb.SequenceAdder() for more details.

  Args:
    observations: [L, ...] Tensor.
    actions: [L, ...] Tensor.
    rewards: [L] Tensor.
    discounts: [L] Tensor.
    extra_spec: A possibly nested structure of specs for extras. This function
      will generate fake (all-zero) extras.
    period: The period with which we add sequences.
    sequence_length: The fixed length of sequences we wish to add.

  Returns:
    (o_t, a_t, r_t, d_t, e_t) Tuple.
  """

  length = tf.shape(rewards)[0]
  first = tf.random.uniform(shape=(), minval=0, maxval=length, dtype=tf.int32)
  first = first // period * period  # Get a multiple of `period`.
  to = tf.minimum(first + sequence_length, length)

  def _slice_and_pad(x):
    pad_length = sequence_length + first - to
    padding_shape = tf.concat([[pad_length], tf.shape(x)[1:]], axis=0)
    result = tf.concat([x[first:to], tf.zeros(padding_shape, x.dtype)], axis=0)
    result.set_shape([sequence_length] + x.shape.as_list()[1:])
    return result

  o_t = tree.map_structure(_slice_and_pad, observations)
  a_t = tree.map_structure(_slice_and_pad, actions)
  r_t = _slice_and_pad(rewards)
  d_t = _slice_and_pad(discounts)

  def _sequence_zeros(spec):
    return tf.zeros([sequence_length] + spec.shape, spec.dtype)

  e_t = tree.map_structure(_sequence_zeros, extra_spec)

  key = tf.zeros([sequence_length], tf.uint64)
  probability = tf.ones([sequence_length], tf.float64)
  table_size = tf.ones([sequence_length], tf.int64)
  info = reverb.SampleInfo(
      key=key, probability=probability, table_size=table_size)
  return reverb.ReplaySample(info=info, data=(o_t, a_t, r_t, d_t, e_t))
