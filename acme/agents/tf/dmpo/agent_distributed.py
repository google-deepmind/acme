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

"""Defines the distributional MPO distributed agent class."""

from typing import Callable, Dict, Optional, Sequence

import acme
from acme import datasets
from acme import specs
from acme import types
from acme.adders import reverb as adders
from acme.agents.tf import actors
from acme.agents.tf.dmpo import learning
from acme.datasets import image_augmentation
from acme.tf import networks
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
from acme.utils import counting
from acme.utils import loggers
from acme.utils import lp_utils
from acme.utils import observers as observers_lib
import dm_env
import launchpad as lp
import reverb
import sonnet as snt
import tensorflow as tf


class DistributedDistributionalMPO:
  """Program definition for distributional MPO."""

  def __init__(
      self,
      environment_factory: Callable[[bool], dm_env.Environment],
      network_factory: Callable[[specs.BoundedArray], Dict[str, snt.Module]],
      num_actors: int = 1,
      num_caches: int = 0,
      environment_spec: Optional[specs.EnvironmentSpec] = None,
      batch_size: int = 256,
      prefetch_size: int = 4,
      observation_augmentation: Optional[types.TensorTransformation] = None,
      min_replay_size: int = 1000,
      max_replay_size: int = 1000000,
      samples_per_insert: Optional[float] = 32.0,
      n_step: int = 5,
      num_samples: int = 20,
      additional_discount: float = 0.99,
      target_policy_update_period: int = 100,
      target_critic_update_period: int = 100,
      variable_update_period: int = 1000,
      policy_loss_factory: Optional[Callable[[], snt.Module]] = None,
      max_actor_steps: Optional[int] = None,
      log_every: float = 10.0,
      make_observers: Optional[Callable[
          [], Sequence[observers_lib.EnvLoopObserver]]] = None):

    if environment_spec is None:
      environment_spec = specs.make_environment_spec(environment_factory(False))

    self._environment_factory = environment_factory
    self._network_factory = network_factory
    self._policy_loss_factory = policy_loss_factory
    self._environment_spec = environment_spec
    self._num_actors = num_actors
    self._num_caches = num_caches
    self._batch_size = batch_size
    self._prefetch_size = prefetch_size
    self._observation_augmentation = observation_augmentation
    self._min_replay_size = min_replay_size
    self._max_replay_size = max_replay_size
    self._samples_per_insert = samples_per_insert
    self._n_step = n_step
    self._additional_discount = additional_discount
    self._num_samples = num_samples
    self._target_policy_update_period = target_policy_update_period
    self._target_critic_update_period = target_critic_update_period
    self._variable_update_period = variable_update_period
    self._max_actor_steps = max_actor_steps
    self._log_every = log_every
    self._make_observers = make_observers

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

    # Create online and target networks.
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
    dataset = datasets.make_reverb_dataset(server_address=replay.server_address)
    dataset = dataset.batch(self._batch_size, drop_remainder=True)
    if self._observation_augmentation:
      transform = image_augmentation.make_transform(
          observation_transform=self._observation_augmentation)
      dataset = dataset.map(
          transform, num_parallel_calls=16, deterministic=False)
    dataset = dataset.prefetch(self._prefetch_size)

    counter = counting.Counter(counter, 'learner')
    logger = loggers.make_default_logger(
        'learner', time_delta=self._log_every, steps_key='learner_steps')

    # Create policy loss module if a factory is passed.
    if self._policy_loss_factory:
      policy_loss_module = self._policy_loss_factory()
    else:
      policy_loss_module = None

    # Return the learning agent.
    return learning.DistributionalMPOLearner(
        policy_network=online_networks['policy'],
        critic_network=online_networks['critic'],
        observation_network=observation_network,
        target_policy_network=target_networks['policy'],
        target_critic_network=target_networks['critic'],
        target_observation_network=target_observation_network,
        discount=self._additional_discount,
        num_samples=self._num_samples,
        target_policy_update_period=self._target_policy_update_period,
        target_critic_update_period=self._target_critic_update_period,
        policy_loss_module=policy_loss_module,
        dataset=dataset,
        counter=counter,
        logger=logger)

  def actor(
      self,
      replay: reverb.Client,
      variable_source: acme.VariableSource,
      counter: counting.Counter,
      actor_id: int,
  ) -> acme.EnvironmentLoop:
    """The actor process."""

    action_spec = self._environment_spec.actions
    observation_spec = self._environment_spec.observations

    # Create environment and target networks to act with.
    environment = self._environment_factory(False)
    agent_networks = self._network_factory(action_spec)

    # Make sure observation network is defined.
    observation_network = agent_networks.get('observation', tf.identity)

    # Create a stochastic behavior policy.
    behavior_network = snt.Sequential([
        observation_network,
        agent_networks['policy'],
        networks.StochasticSamplingHead(),
    ])

    # Ensure network variables are created.
    tf2_utils.create_variables(behavior_network, [observation_spec])
    policy_variables = {'policy': behavior_network.variables}

    # Create the variable client responsible for keeping the actor up-to-date.
    variable_client = tf2_variable_utils.VariableClient(
        variable_source,
        policy_variables,
        update_period=self._variable_update_period)

    # Make sure not to use a random policy after checkpoint restoration by
    # assigning variables before running the environment loop.
    variable_client.update_and_wait()

    # Component to add things into replay.
    adder = adders.NStepTransitionAdder(
        client=replay,
        n_step=self._n_step,
        discount=self._additional_discount)

    # Create the agent.
    actor = actors.FeedForwardActor(
        policy_network=behavior_network,
        adder=adder,
        variable_client=variable_client)

    # Create logger and counter; only the first actor stores logs to bigtable.
    save_data = actor_id == 0
    counter = counting.Counter(counter, 'actor')
    logger = loggers.make_default_logger(
        'actor',
        save_data=save_data,
        time_delta=self._log_every,
        steps_key='actor_steps')
    observers = self._make_observers() if self._make_observers else ()

    # Create the run loop and return it.
    return acme.EnvironmentLoop(
        environment, actor, counter, logger, observers=observers)

  def evaluator(
      self,
      variable_source: acme.VariableSource,
      counter: counting.Counter,
  ):
    """The evaluation process."""

    action_spec = self._environment_spec.actions
    observation_spec = self._environment_spec.observations

    # Create environment and target networks to act with.
    environment = self._environment_factory(True)
    agent_networks = self._network_factory(action_spec)

    # Make sure observation network is defined.
    observation_network = agent_networks.get('observation', tf.identity)

    # Create a stochastic behavior policy.
    evaluator_network = snt.Sequential([
        observation_network,
        agent_networks['policy'],
        networks.StochasticMeanHead(),
    ])

    # Ensure network variables are created.
    tf2_utils.create_variables(evaluator_network, [observation_spec])
    policy_variables = {'policy': evaluator_network.variables}

    # Create the variable client responsible for keeping the actor up-to-date.
    variable_client = tf2_variable_utils.VariableClient(
        variable_source,
        policy_variables,
        update_period=self._variable_update_period)

    # Make sure not to evaluate a random actor by assigning variables before
    # running the environment loop.
    variable_client.update_and_wait()

    # Create the agent.
    evaluator = actors.FeedForwardActor(
        policy_network=evaluator_network, variable_client=variable_client)

    # Create logger and counter.
    counter = counting.Counter(counter, 'evaluator')
    logger = loggers.make_default_logger(
        'evaluator', time_delta=self._log_every, steps_key='evaluator_steps')
    observers = self._make_observers() if self._make_observers else ()

    # Create the run loop and return it.
    return acme.EnvironmentLoop(
        environment,
        evaluator,
        counter,
        logger,
        observers=observers)

  def build(self, name='dmpo'):
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
        program.add_node(
            lp.CourierNode(self.actor, replay, source, counter, actor_id))

    return program
