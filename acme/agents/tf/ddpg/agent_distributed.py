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

"""Defines the distribted DDPG (D3PG) agent class."""

from typing import Callable, Dict, Optional

import acme
from acme import datasets
from acme import specs
from acme.adders import reverb as adders
from acme.agents.tf import actors
from acme.agents.tf.ddpg import learning
from acme.tf import networks
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
from acme.utils import counting
from acme.utils import loggers
from acme.utils import lp_utils
import dm_env
import launchpad as lp
import reverb
import sonnet as snt
import tensorflow as tf


class DistributedDDPG:
  """Program definition for distributed DDPG (D3PG)."""

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
      n_step: int = 5,
      sigma: float = 0.3,
      clipping: bool = True,
      discount: float = 0.99,
      target_update_period: int = 100,
      variable_update_period: int = 1000,
      max_actor_steps: Optional[int] = None,
      log_every: float = 10.0,
  ):

    if not environment_spec:
      environment_spec = specs.make_environment_spec(environment_factory(False))

    self._environment_factory = environment_factory
    self._network_factory = network_factory
    self._environment_spec = environment_spec
    self._num_actors = num_actors
    self._num_caches = num_caches
    self._batch_size = batch_size
    self._prefetch_size = prefetch_size
    self._min_replay_size = min_replay_size
    self._max_replay_size = max_replay_size
    self._samples_per_insert = samples_per_insert
    self._n_step = n_step
    self._sigma = sigma
    self._clipping = clipping
    self._discount = discount
    self._target_update_period = target_update_period
    self._variable_update_period = variable_update_period
    self._max_actor_steps = max_actor_steps
    self._log_every = log_every

  def replay(self):
    """The replay storage."""
    if self._samples_per_insert is not None:
      # Create enough of an error buffer to give a 10% tolerance in rate.
      samples_per_insert_tolerance = 0.1 * self._samples_per_insert
      error_buffer = self._min_replay_size * samples_per_insert_tolerance

      limiter = reverb.rate_limiters.SampleToInsertRatio(
          min_size_to_sample=self._min_replay_size,
          samples_per_insert=self._samples_per_insert,
          error_buffer=error_buffer)
    else:
      limiter = reverb.rate_limiters.MinSize(self._min_replay_size)
    replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=self._max_replay_size,
        rate_limiter=limiter,
        signature=adders.NStepTransitionAdder.signature(
            self._environment_spec))
    return [replay_table]

  def counter(self):
    return tf2_savers.CheckpointingRunner(counting.Counter(),
                                          time_delta_minutes=1,
                                          subdirectory='counter')

  def coordinator(self, counter: counting.Counter, max_actor_steps: int):
    return lp_utils.StepsLimiter(counter, max_actor_steps)

  def learner(
      self,
      replay: reverb.Client,
      counter: counting.Counter,
  ):
    """The Learning part of the agent."""

    act_spec = self._environment_spec.actions
    obs_spec = self._environment_spec.observations

    # Create the networks to optimize (online) and target networks.
    online_networks = self._network_factory(act_spec)
    target_networks = self._network_factory(act_spec)

    # Make sure observation network is a Sonnet Module.
    observation_network = online_networks.get('observation', tf.identity)
    target_observation_network = target_networks.get('observation', tf.identity)
    observation_network = tf2_utils.to_sonnet_module(observation_network)
    target_observation_network = tf2_utils.to_sonnet_module(
        target_observation_network)

    # Get embedding spec and create observation network variables.
    emb_spec = tf2_utils.create_variables(observation_network, [obs_spec])

    # Create variables.
    tf2_utils.create_variables(online_networks['policy'], [emb_spec])
    tf2_utils.create_variables(online_networks['critic'], [emb_spec, act_spec])
    tf2_utils.create_variables(target_networks['policy'], [emb_spec])
    tf2_utils.create_variables(target_networks['critic'], [emb_spec, act_spec])
    tf2_utils.create_variables(target_observation_network, [obs_spec])

    # The dataset object to learn from.
    dataset = datasets.make_reverb_dataset(
        server_address=replay.server_address,
        batch_size=self._batch_size,
        prefetch_size=self._prefetch_size)

    # Create optimizers.
    policy_optimizer = snt.optimizers.Adam(learning_rate=1e-4)
    critic_optimizer = snt.optimizers.Adam(learning_rate=1e-4)

    counter = counting.Counter(counter, 'learner')
    logger = loggers.make_default_logger(
        'learner', time_delta=self._log_every, steps_key='learner_steps')

    # Return the learning agent.
    return learning.DDPGLearner(
        policy_network=online_networks['policy'],
        critic_network=online_networks['critic'],
        observation_network=observation_network,
        target_policy_network=target_networks['policy'],
        target_critic_network=target_networks['critic'],
        target_observation_network=target_observation_network,
        discount=self._discount,
        target_update_period=self._target_update_period,
        dataset=dataset,
        policy_optimizer=policy_optimizer,
        critic_optimizer=critic_optimizer,
        clipping=self._clipping,
        counter=counter,
        logger=logger,
    )

  def actor(
      self,
      replay: reverb.Client,
      variable_source: acme.VariableSource,
      counter: counting.Counter,
  ):
    """The actor process."""

    action_spec = self._environment_spec.actions
    observation_spec = self._environment_spec.observations

    # Create environment and behavior networks
    environment = self._environment_factory(False)
    agent_networks = self._network_factory(action_spec)

    # Create behavior network by adding some random dithering.
    behavior_network = snt.Sequential([
        agent_networks.get('observation', tf.identity),
        agent_networks.get('policy'),
        networks.ClippedGaussian(self._sigma),
    ])

    # Ensure network variables are created.
    tf2_utils.create_variables(behavior_network, [observation_spec])
    variables = {'policy': behavior_network.variables}

    # Create the variable client responsible for keeping the actor up-to-date.
    variable_client = tf2_variable_utils.VariableClient(
        variable_source, variables, update_period=self._variable_update_period)

    # Make sure not to use a random policy after checkpoint restoration by
    # assigning variables before running the environment loop.
    variable_client.update_and_wait()

    # Component to add things into replay.
    adder = adders.NStepTransitionAdder(
        client=replay, n_step=self._n_step, discount=self._discount)

    # Create the agent.
    actor = actors.FeedForwardActor(
        behavior_network, adder=adder, variable_client=variable_client)

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
  ):
    """The evaluation process."""

    action_spec = self._environment_spec.actions
    observation_spec = self._environment_spec.observations

    # Create environment and evaluator networks
    environment = self._environment_factory(True)
    agent_networks = self._network_factory(action_spec)

    # Create evaluator network.
    evaluator_network = snt.Sequential([
        agent_networks.get('observation', tf.identity),
        agent_networks.get('policy'),
    ])

    # Ensure network variables are created.
    tf2_utils.create_variables(evaluator_network, [observation_spec])
    variables = {'policy': evaluator_network.variables}

    # Create the variable client responsible for keeping the actor up-to-date.
    variable_client = tf2_variable_utils.VariableClient(
        variable_source, variables, update_period=self._variable_update_period)

    # Make sure not to evaluate a random actor by assigning variables before
    # running the environment loop.
    variable_client.update_and_wait()

    # Create the evaluator; note it will not add experience to replay.
    evaluator = actors.FeedForwardActor(
        evaluator_network, variable_client=variable_client)

    # Create logger and counter.
    counter = counting.Counter(counter, 'evaluator')
    logger = loggers.make_default_logger(
        'evaluator', time_delta=self._log_every, steps_key='evaluator_steps')

    # Create the run loop and return it.
    return acme.EnvironmentLoop(
        environment, evaluator, counter, logger)

  def build(self, name='ddpg'):
    """Build the distributed agent topology."""
    program = lp.Program(name=name)

    with program.group('replay'):
      replay = program.add_node(lp.ReverbNode(self.replay))

    with program.group('counter'):
      counter = program.add_node(lp.CourierNode(self.counter))

      if self._max_actor_steps:
        _ = program.add_node(
            lp.CourierNode(self.coordinator, counter, self._max_actor_steps))

    with program.group('learner'):
      learner = program.add_node(
          lp.CourierNode(self.learner, replay, counter))

    with program.group('evaluator'):
      program.add_node(
          lp.CourierNode(self.evaluator, learner, counter))

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
