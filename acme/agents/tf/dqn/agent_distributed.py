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

"""Defines the DQN agent class."""

import copy
from typing import Callable, Optional

import acme
from acme import datasets
from acme import specs
from acme.adders import reverb as adders
from acme.agents.tf import actors
from acme.agents.tf.dqn import learning
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
from acme.utils import counting
from acme.utils import loggers
from acme.utils import lp_utils
import dm_env
import launchpad as lp
import numpy as np
import reverb
import sonnet as snt
import trfl


class DistributedDQN:
  """Distributed DQN agent."""

  def __init__(
      self,
      environment_factory: Callable[[bool], dm_env.Environment],
      network_factory: Callable[[specs.DiscreteArray], snt.Module],
      num_actors: int,
      num_caches: int = 1,
      batch_size: int = 256,
      prefetch_size: int = 4,
      target_update_period: int = 100,
      samples_per_insert: float = 32.0,
      min_replay_size: int = 1000,
      max_replay_size: int = 1_000_000,
      importance_sampling_exponent: float = 0.2,
      priority_exponent: float = 0.6,
      n_step: int = 5,
      learning_rate: float = 1e-3,
      evaluator_epsilon: float = 0.,
      max_actor_steps: Optional[int] = None,
      discount: float = 0.99,
      environment_spec: Optional[specs.EnvironmentSpec] = None,
      variable_update_period: int = 1000,
  ):

    assert num_caches >= 1

    if environment_spec is None:
      environment_spec = specs.make_environment_spec(environment_factory(False))

    self._environment_factory = environment_factory
    self._network_factory = network_factory
    self._num_actors = num_actors
    self._num_caches = num_caches
    self._env_spec = environment_spec
    self._batch_size = batch_size
    self._prefetch_size = prefetch_size
    self._target_update_period = target_update_period
    self._samples_per_insert = samples_per_insert
    self._min_replay_size = min_replay_size
    self._max_replay_size = max_replay_size
    self._importance_sampling_exponent = importance_sampling_exponent
    self._priority_exponent = priority_exponent
    self._n_step = n_step
    self._learning_rate = learning_rate
    self._evaluator_epsilon = evaluator_epsilon
    self._max_actor_steps = max_actor_steps
    self._discount = discount
    self._variable_update_period = variable_update_period

  def replay(self):
    """The replay storage."""
    if self._samples_per_insert:
      limiter = reverb.rate_limiters.SampleToInsertRatio(
          min_size_to_sample=self._min_replay_size,
          samples_per_insert=self._samples_per_insert,
          error_buffer=self._batch_size)
    else:
      limiter = reverb.rate_limiters.MinSize(self._min_replay_size)
    replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Prioritized(self._priority_exponent),
        remover=reverb.selectors.Fifo(),
        max_size=self._max_replay_size,
        rate_limiter=limiter,
        signature=adders.NStepTransitionAdder.signature(self._env_spec))
    return [replay_table]

  def counter(self):
    """Creates the master counter process."""
    return tf2_savers.CheckpointingRunner(
        counting.Counter(), time_delta_minutes=1, subdirectory='counter')

  def coordinator(self, counter: counting.Counter, max_actor_steps: int):
    return lp_utils.StepsLimiter(counter, max_actor_steps)

  def learner(self, replay: reverb.Client, counter: counting.Counter):
    """The Learning part of the agent."""

    # Create the networks.
    network = self._network_factory(self._env_spec.actions)
    target_network = copy.deepcopy(network)

    tf2_utils.create_variables(network, [self._env_spec.observations])
    tf2_utils.create_variables(target_network, [self._env_spec.observations])

    # The dataset object to learn from.
    replay_client = reverb.Client(replay.server_address)
    dataset = datasets.make_reverb_dataset(
        server_address=replay.server_address,
        batch_size=self._batch_size,
        prefetch_size=self._prefetch_size)

    logger = loggers.make_default_logger('learner', steps_key='learner_steps')

    # Return the learning agent.
    counter = counting.Counter(counter, 'learner')

    learner = learning.DQNLearner(
        network=network,
        target_network=target_network,
        discount=self._discount,
        importance_sampling_exponent=self._importance_sampling_exponent,
        learning_rate=self._learning_rate,
        target_update_period=self._target_update_period,
        dataset=dataset,
        replay_client=replay_client,
        counter=counter,
        logger=logger)
    return tf2_savers.CheckpointingRunner(
        learner, subdirectory='dqn_learner', time_delta_minutes=60)

  def actor(
      self,
      replay: reverb.Client,
      variable_source: acme.VariableSource,
      counter: counting.Counter,
      epsilon: float,
  ) -> acme.EnvironmentLoop:
    """The actor process."""
    environment = self._environment_factory(False)
    network = self._network_factory(self._env_spec.actions)

    # Just inline the policy network here.
    policy_network = snt.Sequential([
        network,
        lambda q: trfl.epsilon_greedy(q, epsilon=epsilon).sample(),
    ])

    tf2_utils.create_variables(policy_network, [self._env_spec.observations])
    variable_client = tf2_variable_utils.VariableClient(
        client=variable_source,
        variables={'policy': policy_network.trainable_variables},
        update_period=self._variable_update_period)

    # Make sure not to use a random policy after checkpoint restoration by
    # assigning variables before running the environment loop.
    variable_client.update_and_wait()

    # Component to add things into replay.
    adder = adders.NStepTransitionAdder(
        client=replay,
        n_step=self._n_step,
        discount=self._discount,
    )

    # Create the agent.
    actor = actors.FeedForwardActor(policy_network, adder, variable_client)

    # Create the loop to connect environment and agent.
    counter = counting.Counter(counter, 'actor')
    logger = loggers.make_default_logger(
        'actor', save_data=False, steps_key='actor_steps')
    return acme.EnvironmentLoop(environment, actor, counter, logger)

  def evaluator(
      self,
      variable_source: acme.VariableSource,
      counter: counting.Counter,
  ):
    """The evaluation process."""
    environment = self._environment_factory(True)
    network = self._network_factory(self._env_spec.actions)

    # Just inline the policy network here.
    policy_network = snt.Sequential([
        network,
        lambda q: trfl.epsilon_greedy(q, self._evaluator_epsilon).sample(),
    ])

    tf2_utils.create_variables(policy_network, [self._env_spec.observations])

    variable_client = tf2_variable_utils.VariableClient(
        client=variable_source,
        variables={'policy': policy_network.trainable_variables},
        update_period=self._variable_update_period)

    # Make sure not to use a random policy after checkpoint restoration by
    # assigning variables before running the environment loop.
    variable_client.update_and_wait()

    # Create the agent.
    actor = actors.FeedForwardActor(
        policy_network, variable_client=variable_client)

    # Create the run loop and return it.
    logger = loggers.make_default_logger(
        'evaluator', steps_key='evaluator_steps')
    counter = counting.Counter(counter, 'evaluator')
    return acme.EnvironmentLoop(
        environment, actor, counter=counter, logger=logger)

  def build(self, name='dqn'):
    """Build the distributed agent topology."""
    program = lp.Program(name=name)

    with program.group('replay'):
      replay = program.add_node(lp.ReverbNode(self.replay))

    with program.group('counter'):
      counter = program.add_node(lp.CourierNode(self.counter))

      if self._max_actor_steps:
        program.add_node(
            lp.CourierNode(self.coordinator, counter, self._max_actor_steps))

    with program.group('learner'):
      learner = program.add_node(lp.CourierNode(self.learner, replay, counter))

    with program.group('evaluator'):
      program.add_node(lp.CourierNode(self.evaluator, learner, counter))

    # Generate an epsilon for each actor.
    epsilons = np.flip(np.logspace(1, 8, self._num_actors, base=0.4), axis=0)

    with program.group('cacher'):
      # Create a set of learner caches.
      sources = []
      for _ in range(self._num_caches):
        cacher = program.add_node(
            lp.CacherNode(
                learner, refresh_interval_ms=2000, stale_after_ms=4000))
        sources.append(cacher)

    with program.group('actor'):
      # Add actors which pull round-robin from our variable sources.
      for actor_id, epsilon in enumerate(epsilons):
        source = sources[actor_id % len(sources)]
        program.add_node(
            lp.CourierNode(self.actor, replay, source, counter, epsilon))

    return program
