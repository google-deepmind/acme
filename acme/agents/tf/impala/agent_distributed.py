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

"""Defines the IMPALA Launchpad program."""

from typing import Callable, Optional

import acme
from acme import datasets
from acme import specs
from acme.adders import reverb as adders
from acme.agents.tf.impala import acting
from acme.agents.tf.impala import learning
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
from acme.utils import counting
from acme.utils import loggers
import dm_env
import launchpad as lp
import reverb
import sonnet as snt
import tensorflow as tf


class DistributedIMPALA:
  """Program definition for IMPALA."""

  def __init__(self,
               environment_factory: Callable[[bool], dm_env.Environment],
               network_factory: Callable[[specs.DiscreteArray], snt.RNNCore],
               num_actors: int,
               sequence_length: int,
               sequence_period: int,
               environment_spec: Optional[specs.EnvironmentSpec] = None,
               batch_size: int = 256,
               prefetch_size: int = 4,
               max_queue_size: int = 10_000,
               learning_rate: float = 1e-3,
               discount: float = 0.99,
               entropy_cost: float = 0.01,
               baseline_cost: float = 0.5,
               max_abs_reward: Optional[float] = None,
               max_gradient_norm: Optional[float] = None,
               variable_update_period: int = 1000,
               save_logs: bool = False):

    if environment_spec is None:
      environment_spec = specs.make_environment_spec(environment_factory(False))

    self._environment_factory = environment_factory
    self._network_factory = network_factory
    self._environment_spec = environment_spec
    self._num_actors = num_actors
    self._batch_size = batch_size
    self._prefetch_size = prefetch_size
    self._sequence_length = sequence_length
    self._max_queue_size = max_queue_size
    self._sequence_period = sequence_period
    self._discount = discount
    self._learning_rate = learning_rate
    self._entropy_cost = entropy_cost
    self._baseline_cost = baseline_cost
    self._max_abs_reward = max_abs_reward
    self._max_gradient_norm = max_gradient_norm
    self._variable_update_period = variable_update_period
    self._save_logs = save_logs

  def queue(self):
    """The queue."""
    num_actions = self._environment_spec.actions.num_values
    network = self._network_factory(self._environment_spec.actions)
    extra_spec = {
        'core_state': network.initial_state(1),
        'logits': tf.ones(shape=(1, num_actions), dtype=tf.float32)
    }
    # Remove batch dimensions.
    extra_spec = tf2_utils.squeeze_batch_dim(extra_spec)
    signature = adders.SequenceAdder.signature(
        self._environment_spec,
        extra_spec,
        sequence_length=self._sequence_length)
    queue = reverb.Table.queue(
        name=adders.DEFAULT_PRIORITY_TABLE,
        max_size=self._max_queue_size,
        signature=signature)
    return [queue]

  def counter(self):
    """Creates the master counter process."""
    return tf2_savers.CheckpointingRunner(
        counting.Counter(), time_delta_minutes=1, subdirectory='counter')

  def learner(self, queue: reverb.Client, counter: counting.Counter):
    """The Learning part of the agent."""
    # Use architect and create the environment.
    # Create the networks.
    network = self._network_factory(self._environment_spec.actions)
    tf2_utils.create_variables(network, [self._environment_spec.observations])

    # The dataset object to learn from.
    dataset = datasets.make_reverb_dataset(
        server_address=queue.server_address,
        batch_size=self._batch_size,
        prefetch_size=self._prefetch_size)

    logger = loggers.make_default_logger('learner', steps_key='learner_steps')
    counter = counting.Counter(counter, 'learner')

    # Return the learning agent.
    learner = learning.IMPALALearner(
        environment_spec=self._environment_spec,
        network=network,
        dataset=dataset,
        discount=self._discount,
        learning_rate=self._learning_rate,
        entropy_cost=self._entropy_cost,
        baseline_cost=self._baseline_cost,
        max_abs_reward=self._max_abs_reward,
        max_gradient_norm=self._max_gradient_norm,
        counter=counter,
        logger=logger,
    )

    return tf2_savers.CheckpointingRunner(learner,
                                          time_delta_minutes=5,
                                          subdirectory='impala_learner')

  def actor(
      self,
      replay: reverb.Client,
      variable_source: acme.VariableSource,
      counter: counting.Counter,
  ) -> acme.EnvironmentLoop:
    """The actor process."""
    environment = self._environment_factory(False)
    network = self._network_factory(self._environment_spec.actions)
    tf2_utils.create_variables(network, [self._environment_spec.observations])

    # Component to add things into the queue.
    adder = adders.SequenceAdder(
        client=replay,
        period=self._sequence_period,
        sequence_length=self._sequence_length)

    variable_client = tf2_variable_utils.VariableClient(
        client=variable_source,
        variables={'policy': network.variables},
        update_period=self._variable_update_period)

    # Make sure not to use a random policy after checkpoint restoration by
    # assigning variables before running the environment loop.
    variable_client.update_and_wait()

    # Create the agent.
    actor = acting.IMPALAActor(
        network=network,
        variable_client=variable_client,
        adder=adder)

    counter = counting.Counter(counter, 'actor')
    logger = loggers.make_default_logger(
        'actor', save_data=False, steps_key='actor_steps')

    # Create the loop to connect environment and agent.
    return acme.EnvironmentLoop(environment, actor, counter, logger)

  def evaluator(self, variable_source: acme.VariableSource,
                counter: counting.Counter):
    """The evaluation process."""
    environment = self._environment_factory(True)
    network = self._network_factory(self._environment_spec.actions)
    tf2_utils.create_variables(network, [self._environment_spec.observations])

    variable_client = tf2_variable_utils.VariableClient(
        client=variable_source,
        variables={'policy': network.variables},
        update_period=self._variable_update_period)

    # Make sure not to use a random policy after checkpoint restoration by
    # assigning variables before running the environment loop.
    variable_client.update_and_wait()

    # Create the agent.
    actor = acting.IMPALAActor(
        network=network, variable_client=variable_client)

    # Create the run loop and return it.
    logger = loggers.make_default_logger(
        'evaluator', steps_key='evaluator_steps')
    counter = counting.Counter(counter, 'evaluator')
    return acme.EnvironmentLoop(environment, actor, counter, logger)

  def build(self, name='impala'):
    """Build the distributed agent topology."""
    program = lp.Program(name=name)

    with program.group('replay'):
      queue = program.add_node(lp.ReverbNode(self.queue))

    with program.group('counter'):
      counter = program.add_node(lp.CourierNode(self.counter))

    with program.group('learner'):
      learner = program.add_node(
          lp.CourierNode(self.learner, queue, counter))

    with program.group('evaluator'):
      program.add_node(lp.CourierNode(self.evaluator, learner, counter))

    with program.group('cacher'):
      cacher = program.add_node(
          lp.CacherNode(learner, refresh_interval_ms=2000, stale_after_ms=4000))

    with program.group('actor'):
      for _ in range(self._num_actors):
        program.add_node(lp.CourierNode(self.actor, queue, cacher, counter))

    return program
