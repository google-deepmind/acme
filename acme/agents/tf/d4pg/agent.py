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

"""D4PG agent implementation."""

import copy
import dataclasses
import functools
from typing import Iterator, List, Optional, Tuple, Union, Sequence

from acme import adders
from acme import core
from acme import datasets
from acme import specs
from acme import types
from acme.adders import reverb as reverb_adders
from acme.agents import agent
from acme.agents.tf import actors
from acme.agents.tf.d4pg import learning
from acme.tf import networks as network_utils
from acme.tf import utils
from acme.tf import variable_utils
from acme.utils import counting
from acme.utils import loggers
import reverb
import sonnet as snt
import tensorflow as tf

Replicator = Union[snt.distribute.Replicator, snt.distribute.TpuReplicator]


@dataclasses.dataclass
class D4PGConfig:
  """Configuration options for the D4PG agent."""

  accelerator: Optional[str] = None
  discount: float = 0.99
  batch_size: int = 256
  prefetch_size: int = 4
  target_update_period: int = 100
  variable_update_period: int = 1000
  policy_optimizer: Optional[snt.Optimizer] = None
  critic_optimizer: Optional[snt.Optimizer] = None
  min_replay_size: int = 1000
  max_replay_size: int = 1000000
  samples_per_insert: Optional[float] = 32.0
  n_step: int = 5
  sigma: float = 0.3
  clipping: bool = True
  replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE


@dataclasses.dataclass
class D4PGNetworks:
  """Structure containing the networks for D4PG."""

  policy_network: snt.Module
  critic_network: snt.Module
  observation_network: snt.Module

  def __init__(
      self,
      policy_network: snt.Module,
      critic_network: snt.Module,
      observation_network: types.TensorTransformation,
  ):
    # This method is implemented (rather than added by the dataclass decorator)
    # in order to allow observation network to be passed as an arbitrary tensor
    # transformation rather than as a snt Module.
    # TODO(mwhoffman): use Protocol rather than Module/TensorTransformation.
    self.policy_network = policy_network
    self.critic_network = critic_network
    self.observation_network = utils.to_sonnet_module(observation_network)

  def init(self, environment_spec: specs.EnvironmentSpec):
    """Initialize the networks given an environment spec."""
    # Get observation and action specs.
    act_spec = environment_spec.actions
    obs_spec = environment_spec.observations

    # Create variables for the observation net and, as a side-effect, get a
    # spec describing the embedding space.
    emb_spec = utils.create_variables(self.observation_network, [obs_spec])

    # Create variables for the policy and critic nets.
    _ = utils.create_variables(self.policy_network, [emb_spec])
    _ = utils.create_variables(self.critic_network, [emb_spec, act_spec])

  def make_policy(
      self,
      environment_spec: specs.EnvironmentSpec,
      sigma: float = 0.0,
  ) -> snt.Module:
    """Create a single network which evaluates the policy."""
    # Stack the observation and policy networks.
    stack = [
        self.observation_network,
        self.policy_network,
    ]

    # If a stochastic/non-greedy policy is requested, add Gaussian noise on
    # top to enable a simple form of exploration.
    # TODO(mwhoffman): Refactor this to remove it from the class.
    if sigma > 0.0:
      stack += [
          network_utils.ClippedGaussian(sigma),
          network_utils.ClipToSpec(environment_spec.actions),
      ]

    # Return a network which sequentially evaluates everything in the stack.
    return snt.Sequential(stack)


class D4PGBuilder:
  """Builder for D4PG which constructs individual components of the agent."""

  def __init__(self, config: D4PGConfig):
    self._config = config

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
  ) -> List[reverb.Table]:
    """Create tables to insert data into."""
    if self._config.samples_per_insert is None:
      # We will take a samples_per_insert ratio of None to mean that there is
      # no limit, i.e. this only implies a min size limit.
      limiter = reverb.rate_limiters.MinSize(self._config.min_replay_size)

    else:
      # Create enough of an error buffer to give a 10% tolerance in rate.
      samples_per_insert_tolerance = 0.1 * self._config.samples_per_insert
      error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
      limiter = reverb.rate_limiters.SampleToInsertRatio(
          min_size_to_sample=self._config.min_replay_size,
          samples_per_insert=self._config.samples_per_insert,
          error_buffer=error_buffer)

    replay_table = reverb.Table(
        name=self._config.replay_table_name,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=self._config.max_replay_size,
        rate_limiter=limiter,
        signature=reverb_adders.NStepTransitionAdder.signature(
            environment_spec))

    return [replay_table]

  def make_dataset_iterator(
      self,
      reverb_client: reverb.Client,
  ) -> Iterator[reverb.ReplaySample]:
    """Create a dataset iterator to use for learning/updating the agent."""
    # The dataset provides an interface to sample from replay.
    dataset = datasets.make_reverb_dataset(
        table=self._config.replay_table_name,
        server_address=reverb_client.server_address,
        batch_size=self._config.batch_size,
        prefetch_size=self._config.prefetch_size)

    replicator = get_replicator(self._config.accelerator)
    dataset = replicator.experimental_distribute_dataset(dataset)

    # TODO(b/155086959): Fix type stubs and remove.
    return iter(dataset)  # pytype: disable=wrong-arg-types

  def make_adder(
      self,
      replay_client: reverb.Client,
  ) -> adders.Adder:
    """Create an adder which records data generated by the actor/environment."""
    return reverb_adders.NStepTransitionAdder(
        priority_fns={self._config.replay_table_name: lambda x: 1.},
        client=replay_client,
        n_step=self._config.n_step,
        discount=self._config.discount)

  def make_actor(
      self,
      policy_network: snt.Module,
      adder: Optional[adders.Adder] = None,
      variable_source: Optional[core.VariableSource] = None,
  ):
    """Create an actor instance."""
    if variable_source:
      # Create the variable client responsible for keeping the actor up-to-date.
      variable_client = variable_utils.VariableClient(
          client=variable_source,
          variables={'policy': policy_network.variables},
          update_period=self._config.variable_update_period,
      )

      # Make sure not to use a random policy after checkpoint restoration by
      # assigning variables before running the environment loop.
      variable_client.update_and_wait()

    else:
      variable_client = None

    # Create the actor which defines how we take actions.
    return actors.FeedForwardActor(
        policy_network=policy_network,
        adder=adder,
        variable_client=variable_client,
    )

  def make_learner(
      self,
      networks: Tuple[D4PGNetworks, D4PGNetworks],
      dataset: Iterator[reverb.ReplaySample],
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      checkpoint: bool = False,
  ):
    """Creates an instance of the learner."""
    online_networks, target_networks = networks

    # The learner updates the parameters (and initializes them).
    return learning.D4PGLearner(
        policy_network=online_networks.policy_network,
        critic_network=online_networks.critic_network,
        observation_network=online_networks.observation_network,
        target_policy_network=target_networks.policy_network,
        target_critic_network=target_networks.critic_network,
        target_observation_network=target_networks.observation_network,
        policy_optimizer=self._config.policy_optimizer,
        critic_optimizer=self._config.critic_optimizer,
        clipping=self._config.clipping,
        discount=self._config.discount,
        target_update_period=self._config.target_update_period,
        dataset_iterator=dataset,
        replicator=get_replicator(self._config.accelerator),
        counter=counter,
        logger=logger,
        checkpoint=checkpoint,
    )


class D4PG(agent.Agent):
  """D4PG Agent.

  This implements a single-process D4PG agent. This is an actor-critic algorithm
  that generates data via a behavior policy, inserts N-step transitions into
  a replay buffer, and periodically updates the policy (and as a result the
  behavior) by sampling uniformly from this buffer.
  """

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy_network: snt.Module,
      critic_network: snt.Module,
      observation_network: types.TensorTransformation = tf.identity,
      accelerator: Optional[str] = None,
      discount: float = 0.99,
      batch_size: int = 256,
      prefetch_size: int = 4,
      target_update_period: int = 100,
      policy_optimizer: Optional[snt.Optimizer] = None,
      critic_optimizer: Optional[snt.Optimizer] = None,
      min_replay_size: int = 1000,
      max_replay_size: int = 1000000,
      samples_per_insert: float = 32.0,
      n_step: int = 5,
      sigma: float = 0.3,
      clipping: bool = True,
      replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      checkpoint: bool = True,
  ):
    """Initialize the agent.

    Args:
      environment_spec: description of the actions, observations, etc.
      policy_network: the online (optimized) policy.
      critic_network: the online critic.
      observation_network: optional network to transform the observations before
        they are fed into any network.
      accelerator: 'TPU', 'GPU', or 'CPU'. If omitted, the first available
        accelerator type from ['TPU', 'GPU', 'CPU'] will be selected.
      discount: discount to use for TD updates.
      batch_size: batch size for updates.
      prefetch_size: size to prefetch from replay.
      target_update_period: number of learner steps to perform before updating
        the target networks.
      policy_optimizer: optimizer for the policy network updates.
      critic_optimizer: optimizer for the critic network updates.
      min_replay_size: minimum replay size before updating.
      max_replay_size: maximum replay size.
      samples_per_insert: number of samples to take from replay for every insert
        that is made.
      n_step: number of steps to squash into a single transition.
      sigma: standard deviation of zero-mean, Gaussian exploration noise.
      clipping: whether to clip gradients by global norm.
      replay_table_name: string indicating what name to give the replay table.
      counter: counter object used to keep track of steps.
      logger: logger object to be used by learner.
      checkpoint: boolean indicating whether to checkpoint the learner.
    """
    if not accelerator:
      accelerator = _get_first_available_accelerator_type(['TPU', 'GPU', 'CPU'])

    # Create the Builder object which will internally create agent components.
    builder = D4PGBuilder(
        # TODO(mwhoffman): pass the config dataclass in directly.
        # TODO(mwhoffman): use the limiter rather than the workaround below.
        # Right now this modifies min_replay_size and samples_per_insert so that
        # they are not controlled by a limiter and are instead handled by the
        # Agent base class (the above TODO directly references this behavior).
        D4PGConfig(
            accelerator=accelerator,
            discount=discount,
            batch_size=batch_size,
            prefetch_size=prefetch_size,
            target_update_period=target_update_period,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            min_replay_size=1,  # Let the Agent class handle this.
            max_replay_size=max_replay_size,
            samples_per_insert=None,  # Let the Agent class handle this.
            n_step=n_step,
            sigma=sigma,
            clipping=clipping,
            replay_table_name=replay_table_name,
        ))

    replicator = get_replicator(accelerator)

    with replicator.scope():
      # TODO(mwhoffman): pass the network dataclass in directly.
      online_networks = D4PGNetworks(policy_network=policy_network,
                                     critic_network=critic_network,
                                     observation_network=observation_network)

      # Target networks are just a copy of the online networks.
      target_networks = copy.deepcopy(online_networks)

      # Initialize the networks.
      online_networks.init(environment_spec)
      target_networks.init(environment_spec)

    # TODO(mwhoffman): either make this Dataclass or pass only one struct.
    # The network struct passed to make_learner is just a tuple for the
    # time-being (for backwards compatibility).
    networks = (online_networks, target_networks)

    # Create the behavior policy.
    policy_network = online_networks.make_policy(environment_spec, sigma)

    # Create the replay server and grab its address.
    replay_tables = builder.make_replay_tables(environment_spec)
    replay_server = reverb.Server(replay_tables, port=None)
    replay_client = reverb.Client(f'localhost:{replay_server.port}')

    # Create actor, dataset, and learner for generating, storing, and consuming
    # data respectively.
    adder = builder.make_adder(replay_client)
    actor = builder.make_actor(policy_network, adder)
    dataset = builder.make_dataset_iterator(replay_client)
    learner = builder.make_learner(networks, dataset, counter, logger,
                                   checkpoint)

    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=max(batch_size, min_replay_size),
        observations_per_step=float(batch_size) / samples_per_insert)

    # Save the replay so we don't garbage collect it.
    self._replay_server = replay_server


def _ensure_accelerator(accelerator: str) -> str:
  """Checks for the existence of the expected accelerator type.

  Args:
    accelerator: 'CPU', 'GPU' or 'TPU'.

  Returns:
    The validated `accelerator` argument.

  Raises:
    RuntimeError: Thrown if the expected accelerator isn't found.
  """
  devices = tf.config.get_visible_devices(device_type=accelerator)

  if devices:
    return accelerator
  else:
    error_messages = [f'Couldn\'t find any {accelerator} devices.',
                      'tf.config.get_visible_devices() returned:']
    error_messages.extend([str(d) for d in devices])
    raise RuntimeError('\n'.join(error_messages))


def _get_first_available_accelerator_type(
    wishlist: Sequence[str] = ('TPU', 'GPU', 'CPU')) -> str:
  """Returns the first available accelerator type listed in a wishlist.

  Args:
    wishlist: A sequence of elements from {'CPU', 'GPU', 'TPU'}, listed in
      order of descending preference.

  Returns:
    The first available accelerator type from `wishlist`.

  Raises:
    RuntimeError: Thrown if no accelerators from the `wishlist` are found.
  """
  get_visible_devices = tf.config.get_visible_devices

  for wishlist_device in wishlist:
    devices = get_visible_devices(device_type=wishlist_device)
    if devices:
      return wishlist_device

  available = ', '.join(
      sorted(frozenset([d.type for d in get_visible_devices()])))
  raise RuntimeError(
      'Couldn\'t find any devices from {wishlist}.' +
      f'Only the following types are available: {available}.')


# Only instantiate one replicator per (process, accelerator type), in case
# a replicator stores state that needs to be carried between its method calls.
@functools.lru_cache()
def get_replicator(accelerator: Optional[str]) -> Replicator:
  """Returns a replicator instance appropriate for the given accelerator.

  This caches the instance using functools.cache, so that only one replicator
  is instantiated per process and argument value.

  Args:
    accelerator: None, 'TPU', 'GPU', or 'CPU'. If None, the first available
      accelerator type will be chosen from ('TPU', 'GPU', 'CPU').

  Returns:
    A replicator, for replciating weights, datasets, and updates across
    one or more accelerators.
  """
  if accelerator:
    accelerator = _ensure_accelerator(accelerator)
  else:
    accelerator = _get_first_available_accelerator_type()

  if accelerator == 'TPU':
    tf.tpu.experimental.initialize_tpu_system()
    return snt.distribute.TpuReplicator()
  else:
    return snt.distribute.Replicator()
