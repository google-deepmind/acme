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

"""Defines the SVG0 agent class."""

import copy
from typing import Callable, Dict, Optional

import acme
from acme import specs
from acme.agents.tf.svg0_prior import agent
from acme.tf import savers as tf2_savers
from acme.utils import counting
from acme.utils import loggers
from acme.utils import lp_utils
import dm_env
import launchpad as lp
import reverb
import sonnet as snt


class DistributedSVG0:
  """Program definition for SVG0."""

  def __init__(
      self,
      environment_factory: Callable[[bool], dm_env.Environment],
      network_factory: Callable[[specs.BoundedArray], Dict[str, snt.Module]],
      num_actors: int = 1,
      num_caches: int = 0,
      environment_spec: Optional[specs.EnvironmentSpec] = None,
      batch_size: int = 256,
      prefetch_size: int = 4,
      min_replay_size: int = 1000,
      max_replay_size: int = 1000000,
      samples_per_insert: Optional[float] = 32.0,
      sequence_length: int = 10,
      sigma: float = 0.3,
      discount: float = 0.99,
      policy_optimizer: Optional[snt.Optimizer] = None,
      critic_optimizer: Optional[snt.Optimizer] = None,
      prior_optimizer: Optional[snt.Optimizer] = None,
      distillation_cost: Optional[float] = 1e-3,
      entropy_regularizer_cost: Optional[float] = 1e-3,
      target_update_period: int = 100,
      max_actor_steps: Optional[int] = None,
      log_every: float = 10.0,
  ):

    if not environment_spec:
      environment_spec = specs.make_environment_spec(environment_factory(False))

    # TODO(mwhoffman): Make network_factory directly return the struct.
    # TODO(mwhoffman): Make the factory take the entire spec.
    def wrapped_network_factory(action_spec):
      networks_dict = network_factory(action_spec)
      networks = agent.SVG0Networks(
          policy_network=networks_dict.get('policy'),
          critic_network=networks_dict.get('critic'),
          prior_network=networks_dict.get('prior', None),)
      return networks

    self._environment_factory = environment_factory
    self._network_factory = wrapped_network_factory
    self._environment_spec = environment_spec
    self._sigma = sigma
    self._num_actors = num_actors
    self._num_caches = num_caches
    self._max_actor_steps = max_actor_steps
    self._log_every = log_every
    self._sequence_length = sequence_length

    self._builder = agent.SVG0Builder(
        # TODO(mwhoffman): pass the config dataclass in directly.
        # TODO(mwhoffman): use the limiter rather than the workaround below.
        agent.SVG0Config(
            discount=discount,
            batch_size=batch_size,
            prefetch_size=prefetch_size,
            target_update_period=target_update_period,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            prior_optimizer=prior_optimizer,
            min_replay_size=min_replay_size,
            max_replay_size=max_replay_size,
            samples_per_insert=samples_per_insert,
            sequence_length=sequence_length,
            sigma=sigma,
            distillation_cost=distillation_cost,
            entropy_regularizer_cost=entropy_regularizer_cost,
        ))

  def replay(self):
    """The replay storage."""
    return self._builder.make_replay_tables(self._environment_spec,
                                            self._sequence_length)

  def counter(self):
    return tf2_savers.CheckpointingRunner(counting.Counter(),
                                          time_delta_minutes=1,
                                          subdirectory='counter')

  def coordinator(self, counter: counting.Counter):
    return lp_utils.StepsLimiter(counter, self._max_actor_steps)

  def learner(
      self,
      replay: reverb.Client,
      counter: counting.Counter,
  ):
    """The Learning part of the agent."""

    # Create the networks to optimize (online) and target networks.
    online_networks = self._network_factory(self._environment_spec.actions)
    target_networks = copy.deepcopy(online_networks)

    # Initialize the networks.
    online_networks.init(self._environment_spec)
    target_networks.init(self._environment_spec)

    dataset = self._builder.make_dataset_iterator(replay)
    counter = counting.Counter(counter, 'learner')
    logger = loggers.make_default_logger(
        'learner', time_delta=self._log_every, steps_key='learner_steps')

    return self._builder.make_learner(
        networks=(online_networks, target_networks),
        dataset=dataset,
        counter=counter,
        logger=logger,
    )

  def actor(
      self,
      replay: reverb.Client,
      variable_source: acme.VariableSource,
      counter: counting.Counter,
  ) -> acme.EnvironmentLoop:
    """The actor process."""

    # Create the behavior policy.
    networks = self._network_factory(self._environment_spec.actions)
    networks.init(self._environment_spec)
    policy_network = networks.make_policy()

    # Create the agent.
    actor = self._builder.make_actor(
        policy_network=policy_network,
        adder=self._builder.make_adder(replay),
        variable_source=variable_source,
    )

    # Create the environment.
    environment = self._environment_factory(False)

    # Create logger and counter; actors will not spam bigtable.
    counter = counting.Counter(counter, 'actor')
    logger = loggers.make_default_logger(
        'actor',
        save_data=False,
        time_delta=self._log_every,
        steps_key='actor_steps')

    # Create the loop to connect environment and agent.
    return acme.EnvironmentLoop(environment, actor, counter, logger)

  def evaluator(
      self,
      variable_source: acme.VariableSource,
      counter: counting.Counter,
      logger: Optional[loggers.Logger] = None,
  ):
    """The evaluation process."""

    # Create the behavior policy.
    networks = self._network_factory(self._environment_spec.actions)
    networks.init(self._environment_spec)
    policy_network = networks.make_policy()

    # Create the agent.
    actor = self._builder.make_actor(
        policy_network=policy_network,
        variable_source=variable_source,
        deterministic_policy=True,
    )

    # Make the environment.
    environment = self._environment_factory(True)

    # Create logger and counter.
    counter = counting.Counter(counter, 'evaluator')
    logger = logger or loggers.make_default_logger(
        'evaluator',
        time_delta=self._log_every,
        steps_key='evaluator_steps',
    )

    # Create the run loop and return it.
    return acme.EnvironmentLoop(environment, actor, counter, logger)

  def build(self, name='svg0'):
    """Build the distributed agent topology."""
    program = lp.Program(name=name)

    with program.group('replay'):
      replay = program.add_node(lp.ReverbNode(self.replay))

    with program.group('counter'):
      counter = program.add_node(lp.CourierNode(self.counter))

    if self._max_actor_steps:
      with program.group('coordinator'):
        _ = program.add_node(lp.CourierNode(self.coordinator, counter))

    with program.group('learner'):
      learner = program.add_node(lp.CourierNode(self.learner, replay, counter))

    with program.group('evaluator'):
      program.add_node(lp.CourierNode(self.evaluator, learner, counter))

    if not self._num_caches:
      # Use our learner as a single variable source.
      sources = [learner]
    else:
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
      for actor_id in range(self._num_actors):
        source = sources[actor_id % len(sources)]
        program.add_node(lp.CourierNode(self.actor, replay, source, counter))

    return program
