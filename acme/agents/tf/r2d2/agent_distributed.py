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

"""Defines the Recurrent DQN Launchpad program."""

import copy
from typing import Callable, List, Optional

import acme
from acme import datasets
from acme import specs
from acme.adders import reverb as adders
from acme.agents.tf import actors
from acme.agents.tf.r2d2 import learning
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
from acme.utils import counting
from acme.utils import loggers
import dm_env
import launchpad as lp
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import trfl


class DistributedR2D2:
  """Program definition for Recurrent Replay Distributed DQN (R2D2)."""

  def __init__(self,
               environment_factory: Callable[[bool], dm_env.Environment],
               network_factory: Callable[[specs.DiscreteArray], snt.RNNCore],
               num_actors: int,
               burn_in_length: int,
               trace_length: int,
               replay_period: int,
               environment_spec: Optional[specs.EnvironmentSpec] = None,
               batch_size: int = 256,
               prefetch_size: int = tf.data.experimental.AUTOTUNE,
               min_replay_size: int = 1000,
               max_replay_size: int = 100_000,
               samples_per_insert: float = 32.0,
               discount: float = 0.99,
               priority_exponent: float = 0.6,
               importance_sampling_exponent: float = 0.2,
               variable_update_period: int = 1000,
               learning_rate: float = 1e-3,
               evaluator_epsilon: float = 0.,
               target_update_period: int = 100,
               save_logs: bool = False):

    if environment_spec is None:
      environment_spec = specs.make_environment_spec(environment_factory(False))

    self._environment_factory = environment_factory
    self._network_factory = network_factory
    self._environment_spec = environment_spec
    self._num_actors = num_actors
    self._batch_size = batch_size
    self._prefetch_size = prefetch_size
    self._min_replay_size = min_replay_size
    self._max_replay_size = max_replay_size
    self._samples_per_insert = samples_per_insert
    self._burn_in_length = burn_in_length
    self._trace_length = trace_length
    self._replay_period = replay_period
    self._discount = discount
    self._target_update_period = target_update_period
    self._variable_update_period = variable_update_period
    self._save_logs = save_logs
    self._priority_exponent = priority_exponent
    self._learning_rate = learning_rate
    self._evaluator_epsilon = evaluator_epsilon
    self._importance_sampling_exponent = importance_sampling_exponent

    self._obs_spec = environment_spec.observations

  def replay(self) -> List[reverb.Table]:
    """The replay storage."""
    network = self._network_factory(self._environment_spec.actions)
    extra_spec = {
        'core_state': network.initial_state(1),
    }
    # Remove batch dimensions.
    extra_spec = tf2_utils.squeeze_batch_dim(extra_spec)
    if self._samples_per_insert:
      limiter = reverb.rate_limiters.SampleToInsertRatio(
          min_size_to_sample=self._min_replay_size,
          samples_per_insert=self._samples_per_insert,
          error_buffer=self._batch_size)
    else:
      limiter = reverb.rate_limiters.MinSize(self._min_replay_size)
    table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Prioritized(self._priority_exponent),
        remover=reverb.selectors.Fifo(),
        max_size=self._max_replay_size,
        rate_limiter=limiter,
        signature=adders.SequenceAdder.signature(
            self._environment_spec,
            extra_spec,
            sequence_length=self._burn_in_length + self._trace_length + 1))

    return [table]

  def counter(self):
    """Creates the master counter process."""
    return tf2_savers.CheckpointingRunner(
        counting.Counter(), time_delta_minutes=1, subdirectory='counter')

  def learner(self, replay: reverb.Client, counter: counting.Counter):
    """The Learning part of the agent."""
    # Use architect and create the environment.
    # Create the networks.
    network = self._network_factory(self._environment_spec.actions)
    target_network = copy.deepcopy(network)

    tf2_utils.create_variables(network, [self._obs_spec])
    tf2_utils.create_variables(target_network, [self._obs_spec])

    # The dataset object to learn from.
    reverb_client = reverb.TFClient(replay.server_address)
    sequence_length = self._burn_in_length + self._trace_length + 1
    dataset = datasets.make_reverb_dataset(
        server_address=replay.server_address,
        batch_size=self._batch_size,
        prefetch_size=self._prefetch_size)

    counter = counting.Counter(counter, 'learner')
    logger = loggers.make_default_logger(
        'learner', save_data=True, steps_key='learner_steps')
    # Return the learning agent.
    learner = learning.R2D2Learner(
        environment_spec=self._environment_spec,
        network=network,
        target_network=target_network,
        burn_in_length=self._burn_in_length,
        sequence_length=sequence_length,
        dataset=dataset,
        reverb_client=reverb_client,
        counter=counter,
        logger=logger,
        discount=self._discount,
        target_update_period=self._target_update_period,
        importance_sampling_exponent=self._importance_sampling_exponent,
        learning_rate=self._learning_rate,
        max_replay_size=self._max_replay_size)
    return tf2_savers.CheckpointingRunner(
        wrapped=learner, time_delta_minutes=60, subdirectory='r2d2_learner')

  def actor(
      self,
      replay: reverb.Client,
      variable_source: acme.VariableSource,
      counter: counting.Counter,
      epsilon: float,
  ) -> acme.EnvironmentLoop:
    """The actor process."""
    environment = self._environment_factory(False)
    network = self._network_factory(self._environment_spec.actions)

    tf2_utils.create_variables(network, [self._obs_spec])

    policy_network = snt.DeepRNN([
        network,
        lambda qs: tf.cast(trfl.epsilon_greedy(qs, epsilon).sample(), tf.int32),
    ])

    # Component to add things into replay.
    sequence_length = self._burn_in_length + self._trace_length + 1
    adder = adders.SequenceAdder(
        client=replay,
        period=self._replay_period,
        sequence_length=sequence_length,
        delta_encoded=True,
    )

    variable_client = tf2_variable_utils.VariableClient(
        client=variable_source,
        variables={'policy': policy_network.variables},
        update_period=self._variable_update_period)

    # Make sure not to use a random policy after checkpoint restoration by
    # assigning variables before running the environment loop.
    variable_client.update_and_wait()

    # Create the agent.
    actor = actors.RecurrentActor(
        policy_network=policy_network,
        variable_client=variable_client,
        adder=adder)

    counter = counting.Counter(counter, 'actor')
    logger = loggers.make_default_logger(
        'actor', save_data=False, steps_key='actor_steps')

    # Create the loop to connect environment and agent.
    return acme.EnvironmentLoop(environment, actor, counter, logger)

  def evaluator(
      self,
      variable_source: acme.VariableSource,
      counter: counting.Counter,
  ):
    """The evaluation process."""
    environment = self._environment_factory(True)
    network = self._network_factory(self._environment_spec.actions)

    tf2_utils.create_variables(network, [self._obs_spec])
    policy_network = snt.DeepRNN([
        network,
        lambda qs: tf.cast(tf.argmax(qs, axis=-1), tf.int32),
    ])

    variable_client = tf2_variable_utils.VariableClient(
        client=variable_source,
        variables={'policy': policy_network.variables},
        update_period=self._variable_update_period)

    # Make sure not to use a random policy after checkpoint restoration by
    # assigning variables before running the environment loop.
    variable_client.update_and_wait()

    # Create the agent.
    actor = actors.RecurrentActor(
        policy_network=policy_network, variable_client=variable_client)

    # Create the run loop and return it.
    logger = loggers.make_default_logger(
        'evaluator', save_data=True, steps_key='evaluator_steps')
    counter = counting.Counter(counter, 'evaluator')

    return acme.EnvironmentLoop(environment, actor, counter, logger)

  def build(self, name='r2d2'):
    """Build the distributed agent topology."""
    program = lp.Program(name=name)

    with program.group('replay'):
      replay = program.add_node(lp.ReverbNode(self.replay))

    with program.group('counter'):
      counter = program.add_node(lp.CourierNode(self.counter))

    with program.group('learner'):
      learner = program.add_node(lp.CourierNode(self.learner, replay, counter))

    with program.group('cacher'):
      cacher = program.add_node(
          lp.CacherNode(learner, refresh_interval_ms=2000, stale_after_ms=4000))

    with program.group('evaluator'):
      program.add_node(lp.CourierNode(self.evaluator, cacher, counter))

    # Generate an epsilon for each actor.
    epsilons = np.flip(np.logspace(1, 8, self._num_actors, base=0.4), axis=0)

    with program.group('actor'):
      for epsilon in epsilons:
        program.add_node(
            lp.CourierNode(self.actor, replay, cacher, counter, epsilon))

    return program
