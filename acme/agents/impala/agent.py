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

"""Importance weighted advantage actor-critic (IMPALA) agent implementation."""

from typing import Optional

import acme
from acme import datasets
from acme import specs
from acme import types
from acme.adders import reverb as adders
from acme.agents.impala import acting
from acme.agents.impala import learning
from acme.utils import counting
from acme.utils import loggers
from acme.utils import tf2_utils

import dm_env
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf


class IMPALA(acme.Actor):
  """IMPALA Agent."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      network: snt.RNNCore,
      sequence_length: int,
      sequence_period: int,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
      discount: float = 0.99,
      max_queue_size: int = 100000,
      batch_size: int = 16,
      learning_rate: float = 1e-3,
      entropy_cost: float = 0.01,
      baseline_cost: float = 0.5,
      max_abs_reward: Optional[float] = None,
      max_gradient_norm: Optional[float] = None,
  ):

    num_actions = environment_spec.actions.num_values
    self._logger = logger or loggers.TerminalLogger('agent')
    queue = reverb.Table.queue(
        name=adders.DEFAULT_PRIORITY_TABLE, max_size=max_queue_size)
    self._server = reverb.Server([queue], port=None)
    self._can_sample = lambda: queue.can_sample(batch_size)
    address = f'localhost:{self._server.port}'

    # Component to add things into replay.
    adder = adders.SequenceAdder(
        client=reverb.Client(address),
        period=sequence_period,
        sequence_length=sequence_length,
    )

    # The dataset object to learn from.
    extra_spec = {
        'core_state': network.initial_state(1),
        'logits': tf.ones(shape=(1, num_actions), dtype=tf.float32)
    }
    # Remove batch dimensions.
    extra_spec = tf2_utils.squeeze_batch_dim(extra_spec)
    dataset = datasets.make_reverb_dataset(
        client=reverb.TFClient(address),
        environment_spec=environment_spec,
        batch_size=batch_size,
        extra_spec=extra_spec,
        sequence_length=sequence_length)

    tf2_utils.create_variables(network, [environment_spec.observations])

    self._actor = acting.IMPALAActor(network, adder)
    self._learner = learning.IMPALALearner(
        environment_spec=environment_spec,
        network=network,
        dataset=dataset,
        counter=counter,
        logger=logger,
        discount=discount,
        learning_rate=learning_rate,
        entropy_cost=entropy_cost,
        baseline_cost=baseline_cost,
        max_gradient_norm=max_gradient_norm,
        max_abs_reward=max_abs_reward,
    )

  def observe_first(self, timestep: dm_env.TimeStep):
    self._actor.observe_first(timestep)

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    self._actor.observe(action, next_timestep)

  def update(self):
    # Run a number of learner steps (usually gradient steps).
    while self._can_sample():
      self._learner.step()

  def select_action(self, observation: np.ndarray) -> int:
    return self._actor.select_action(observation)
