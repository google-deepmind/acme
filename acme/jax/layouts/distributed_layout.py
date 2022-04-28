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

"""Program definition for a distributed layout based on a builder."""

import dataclasses
import logging
from typing import Any, Callable, Dict, Optional, Sequence

from acme import core
from acme import environment_loop
from acme import specs
from acme.agents.jax import builders
from acme.jax import inference_server
from acme.jax import networks as networks_lib
from acme.jax import savers
from acme.jax import types
from acme.jax import utils
from acme.jax import variable_utils
from acme.jax import snapshotter
from acme.utils import counting
from acme.utils import loggers
from acme.utils import lp_utils
from acme.utils import observers as observers_lib
import jax
import launchpad as lp
import reverb


ActorId = int
AgentNetwork = Any
PolicyNetwork = Any
NetworkFactory = Callable[[specs.EnvironmentSpec], AgentNetwork]
PolicyFactory = Callable[[AgentNetwork], PolicyNetwork]
MakeActorFn = Callable[[types.PRNGKey, PolicyNetwork, core.VariableSource],
                       core.Actor]
LoggerLabel = str
LoggerStepsKey = str
LoggerFn = Callable[[LoggerLabel, LoggerStepsKey], loggers.Logger]
EvaluatorFactory = Callable[[
    types.PRNGKey,
    core.VariableSource,
    counting.Counter,
    MakeActorFn,
], core.Worker]


SnapshotModelFactory = Callable[
    [AgentNetwork, specs.EnvironmentSpec],
    Dict[str, Callable[[core.VariableSource], types.ModelToSnapshot]]]


def get_default_logger_fn(
    log_to_bigtable: bool = False,
    log_every: float = 10) -> Callable[[ActorId], loggers.Logger]:
  """Creates an actor logger."""

  def create_logger(actor_id: ActorId):
    return loggers.make_default_logger(
        'actor',
        save_data=(log_to_bigtable and actor_id == 0),
        time_delta=log_every,
        steps_key='actor_steps')
  return create_logger


def default_evaluator_factory(
    environment_factory: types.EnvironmentFactory,
    network_factory: NetworkFactory,
    policy_factory: PolicyFactory,
    observers: Sequence[observers_lib.EnvLoopObserver] = (),
    log_to_bigtable: bool = False,
    logger_fn: Optional[LoggerFn] = None) -> EvaluatorFactory:
  """Returns a default evaluator process."""
  def evaluator(
      random_key: networks_lib.PRNGKey,
      variable_source: core.VariableSource,
      counter: counting.Counter,
      make_actor: MakeActorFn,
  ):
    """The evaluation process."""

    # Create environment and evaluator networks
    environment_key, actor_key = jax.random.split(random_key)
    # Environments normally require uint32 as a seed.
    environment = environment_factory(utils.sample_uint32(environment_key))
    networks = network_factory(specs.make_environment_spec(environment))

    actor = make_actor(actor_key, policy_factory(networks), variable_source)

    # Create logger and counter.
    counter = counting.Counter(counter, 'evaluator')
    if logger_fn is not None:
      logger = logger_fn('evaluator', 'actor_steps')
    else:
      logger = loggers.make_default_logger(
          'evaluator', log_to_bigtable, steps_key='actor_steps')

    # Create the run loop and return it.
    return environment_loop.EnvironmentLoop(environment, actor, counter,
                                            logger, observers=observers)
  return evaluator


@dataclasses.dataclass
class CheckpointingConfig:
  """Configuration options for checkpointing.

  Attributes:
    max_to_keep: Maximum number of checkpoints to keep. Does not apply to replay
      checkpointing.
    directory: Where to store the checkpoints.
    add_uid: Whether or not to add a unique identifier, see
      `paths.get_unique_id()` for how it is generated.
    replay_checkpointing_time_delta_minutes: How frequently to write replay
      checkpoints; defaults to None, which disables periodic checkpointing.
      Warning! These are written asynchronously so as not to interrupt other
      replay duties, however this does pose a risk of OOM since items that
      would otherwise be removed are temporarily kept alive for checkpointing
      purposes.
      Note: Since replay buffers tend to be quite large O(100GiB), writing can
      take up to 10 minutes so keep that in mind when setting this frequency.
  """
  max_to_keep: int = 1
  directory: str = '~/acme'
  add_uid: bool = True
  replay_checkpointing_time_delta_minutes: Optional[int] = None


class DistributedLayout:
  """Program definition for a distributed agent based on a builder.

  DEPRECATED: Use make_distributed_program directly.
  """

  def __init__(self,
               seed: int,
               environment_factory: types.EnvironmentFactory,
               network_factory: NetworkFactory,
               builder: builders.GenericActorLearnerBuilder,
               policy_network: PolicyFactory,
               num_actors: int,
               environment_spec: Optional[specs.EnvironmentSpec] = None,
               actor_logger_fn: Optional[Callable[[ActorId],
                                                  loggers.Logger]] = None,
               evaluator_factories: Sequence[EvaluatorFactory] = (),
               device_prefetch: bool = True,
               prefetch_size: int = 1,
               log_to_bigtable: bool = False,
               max_number_of_steps: Optional[int] = None,
               observers: Sequence[observers_lib.EnvLoopObserver] = (),
               multithreading_colocate_learner_and_reverb: bool = False,
               checkpointing_config: Optional[CheckpointingConfig] = None,
               make_snapshot_models: Optional[SnapshotModelFactory] = None,
               inference_server_config: Optional[
                   inference_server.InferenceServerConfig] = None):
    self._seed = seed
    self._environment_factory = environment_factory
    self._network_factory = network_factory
    self._builder = builder
    self._policy_network = policy_network
    self._num_actors = num_actors
    self._environment_spec = environment_spec
    self._actor_logger_fn = actor_logger_fn
    self._evaluator_factories = evaluator_factories
    self._device_prefetch = device_prefetch
    self._prefetch_size = prefetch_size
    self._log_to_bigtable = log_to_bigtable
    self._max_number_of_steps = max_number_of_steps
    self._observers = observers
    self._multithreading_colocate_learner_and_reverb = (
        multithreading_colocate_learner_and_reverb)
    self._checkpointing_config = checkpointing_config
    self._make_snapshot_models = make_snapshot_models
    self._inference_server_config = inference_server_config

  def build(self, name='agent', program: Optional[lp.Program] = None):
    """Build the distributed agent topology."""
    return make_distributed_program(
        self._seed, self._environment_factory, self._network_factory,
        self._builder, self._policy_network, self._num_actors,
        self._environment_spec, self._actor_logger_fn,
        self._evaluator_factories, self._device_prefetch, self._prefetch_size,
        self._log_to_bigtable, self._max_number_of_steps, self._observers,
        self._multithreading_colocate_learner_and_reverb,
        self._checkpointing_config, self._make_snapshot_models,
        self._inference_server_config, name, program)


def make_distributed_program(
    seed: int,
    environment_factory: types.EnvironmentFactory,
    network_factory: NetworkFactory,
    builder: builders.GenericActorLearnerBuilder,
    policy_network_factory: PolicyFactory,
    num_actors: int,
    environment_spec: Optional[specs.EnvironmentSpec] = None,
    actor_logger_fn: Optional[Callable[[ActorId], loggers.Logger]] = None,
    evaluator_factories: Sequence[EvaluatorFactory] = (),
    device_prefetch: bool = True,
    prefetch_size: int = 1,
    log_to_bigtable: bool = False,
    max_number_of_steps: Optional[int] = None,
    observers: Sequence[observers_lib.EnvLoopObserver] = (),
    multithreading_colocate_learner_and_reverb: bool = False,
    checkpointing_config: Optional[CheckpointingConfig] = None,
    make_snapshot_models: Optional[SnapshotModelFactory] = None,
    inference_server_config: Optional[
        inference_server.InferenceServerConfig] = None,
    name='agent',
    program: Optional[lp.Program] = None):
  """Builds distributed agent based on a builder."""

  if prefetch_size < 0:
    raise ValueError(f'Prefetch size={prefetch_size} should be non negative')

  actor_logger_fn = actor_logger_fn or get_default_logger_fn(log_to_bigtable)
  if checkpointing_config is None:
    checkpointing_config = CheckpointingConfig()

  def build_replay():
    """The replay storage."""
    dummy_seed = 1
    spec = (
        environment_spec or
        specs.make_environment_spec(environment_factory(dummy_seed)))
    return builder.make_replay_tables(spec)

  def build_model_saver(variable_source: core.VariableSource):
    environment = environment_factory(0)
    spec = specs.make_environment_spec(environment)
    networks = network_factory(spec)
    models = make_snapshot_models(networks, spec)
    # TODO(raveman): Decouple checkpointing and snahpshotting configs.
    return snapshotter.JAXSnapshotter(
        variable_source=variable_source,
        models=models,
        path=checkpointing_config.directory,
        add_uid=checkpointing_config.add_uid)

  def build_counter():
    return savers.CheckpointingRunner(
        counting.Counter(),
        key='counter',
        subdirectory='counter',
        time_delta_minutes=5,
        directory=checkpointing_config.directory,
        add_uid=checkpointing_config.add_uid,
        max_to_keep=checkpointing_config.max_to_keep)

  def build_learner(
      random_key: networks_lib.PRNGKey,
      replay: reverb.Client,
      counter: counting.Counter,
  ):
    """The Learning part of the agent."""

    iterator = builder.make_dataset_iterator(replay)

    dummy_seed = 1
    spec = (
        environment_spec or
        specs.make_environment_spec(environment_factory(dummy_seed)))

    # Creates the networks to optimize (online) and target networks.
    networks = network_factory(spec)

    if prefetch_size > 1:
      # When working with single GPU we should prefetch to device for
      # efficiency. If running on TPU this isn't necessary as the computation
      # and input placement can be done automatically. For multi-gpu currently
      # the best solution is to pre-fetch to host although this may change in
      # the future.
      device = jax.devices()[0] if device_prefetch else None
      iterator = utils.prefetch(
          iterator, buffer_size=prefetch_size, device=device)
    else:
      logging.info('Not prefetching the iterator.')

    counter = counting.Counter(counter, 'learner')

    learner = builder.make_learner(random_key, networks, iterator, replay,
                                   counter)
    return savers.CheckpointingRunner(
        learner,
        key='learner',
        subdirectory='learner',
        time_delta_minutes=5,
        directory=checkpointing_config.directory,
        add_uid=checkpointing_config.add_uid,
        max_to_keep=checkpointing_config.max_to_keep)

  def build_actor(
      random_key: networks_lib.PRNGKey,
      replay: reverb.Client,
      variable_source: core.VariableSource,
      counter: counting.Counter,
      actor_id: ActorId,
      inference_client: Optional[inference_server.InferenceServer] = None
  ) -> environment_loop.EnvironmentLoop:
    """The actor process."""
    adder = builder.make_adder(replay)

    environment_key, actor_key = jax.random.split(random_key)
    # Create environment and policy core.

    # Environments normally require uint32 as a seed.
    environment = environment_factory(utils.sample_uint32(environment_key))

    if not inference_client:
      networks = network_factory(specs.make_environment_spec(environment))
      policy_network = policy_network_factory(networks)
    else:
      variable_source = variable_utils.ReferenceVariableSource()
      policy_network = inference_client
    actor = builder.make_actor(actor_key, policy_network, adder,
                               variable_source)

    # Create logger and counter.
    counter = counting.Counter(counter, 'actor')
    # Only actor #0 will write to bigtable in order not to spam it too much.
    logger = actor_logger_fn(actor_id)
    # Create the loop to connect environment and agent.
    return environment_loop.EnvironmentLoop(environment, actor, counter,
                                            logger, observers=observers)

  def build_coordinator(counter: counting.Counter, max_actor_steps: int):
    return lp_utils.StepsLimiter(counter, max_actor_steps)

  if not program:
    program = lp.Program(name=name)

  key = jax.random.PRNGKey(seed)

  replay_node = lp.ReverbNode(
      build_replay,
      checkpoint_time_delta_minutes=(
          checkpointing_config.replay_checkpointing_time_delta_minutes))
  replay = replay_node.create_handle()

  counter = program.add_node(lp.CourierNode(build_counter), label='counter')

  if max_number_of_steps is not None:
    program.add_node(
        lp.CourierNode(build_coordinator, counter, max_number_of_steps),
        label='counter')

  learner_key, key = jax.random.split(key)
  learner_node = lp.CourierNode(build_learner, learner_key, replay, counter)
  learner = learner_node.create_handle()

  if multithreading_colocate_learner_and_reverb:
    program.add_node(lp.MultiThreadingColocation([learner_node, replay_node]),
                     label='learner')
  else:
    program.add_node(learner_node, label='learner')
    program.add_node(replay_node, label='replay')

  inference_server_node = None
  if inference_server_config:

    def build_inference_server(random_key: networks_lib.PRNGKey,
                               variable_source: core.VariableSource):
      """Creates an inference server node to be connected to by the actors."""

      # Environments normally require uint32 as a seed.
      environment = environment_factory(random_key)
      networks = network_factory(specs.make_environment_spec(environment))
      policy_network = policy_network_factory(networks)

      if not inference_server_config.batch_size:
        # Inference batch size computation:
        # - In case of 1 inference device it is efficient to use
        #   `batch size == num_envs / 2`, so that inference can execute
        #   in parallel with a subset of environments' steps (it also addresses
        #   the problem of some environments running slower etc.)
        # - In case of multiple inference devices, we just divide the above
        #   batch size.
        # - Batch size can't obviously be smaller than 1.
        inference_server_config.batch_size = max(
            1, num_actors // (2 * len(jax.local_devices())))

      if not inference_server_config.update_period:
        inference_server_config.update_period = (
            1000 * num_actors // inference_server_config.batch_size)

      return inference_server.InferenceServer(
          config=inference_server_config,
          handler=(policy_network
                   if callable(policy_network) else vars(policy_network)),
          variable_source=variable_source,
          devices=jax.local_devices())

    with program.group('inference_server'):
      inference_server_key, key = jax.random.split(key)
      inference_server_node = program.add_node(
          lp.CourierNode(
              build_inference_server,
              inference_server_key,
              learner,
              courier_kwargs={'thread_pool_size': num_actors}))

  def make_actor(random_key: networks_lib.PRNGKey,
                 policy_network: PolicyNetwork,
                 variable_source: core.VariableSource) -> core.Actor:
    return builder.make_actor(
        random_key, policy_network, variable_source=variable_source)

  for evaluator in evaluator_factories:
    evaluator_key, key = jax.random.split(key)
    program.add_node(
        lp.CourierNode(evaluator, evaluator_key, learner, counter,
                       make_actor), label='evaluator')

  for actor_id in range(num_actors):
    actor_key, key = jax.random.split(key)
    program.add_node(
        lp.CourierNode(build_actor, actor_key, replay, learner, counter,
                       actor_id, inference_server_node), label='actor')
  if make_snapshot_models and checkpointing_config:
    program.add_node(lp.CourierNode(build_model_saver, learner),
                     label='model_saver')

  return program
