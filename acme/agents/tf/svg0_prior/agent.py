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

"""SVG0 agent implementation."""

import copy
import dataclasses
from typing import Iterator, List, Optional, Tuple

from acme import adders
from acme import core
from acme import datasets
from acme import specs
from acme.adders import reverb as reverb_adders
from acme.agents import agent
from acme.agents.tf.svg0_prior import acting
from acme.agents.tf.svg0_prior import learning
from acme.tf import utils
from acme.tf import variable_utils
from acme.utils import counting
from acme.utils import loggers
import reverb
import sonnet as snt
import tensorflow as tf


@dataclasses.dataclass
class SVG0Config:
  """Configuration options for the agent."""

  discount: float = 0.99
  batch_size: int = 256
  prefetch_size: int = 4
  target_update_period: int = 100
  policy_optimizer: Optional[snt.Optimizer] = None
  critic_optimizer: Optional[snt.Optimizer] = None
  prior_optimizer: Optional[snt.Optimizer] = None
  min_replay_size: int = 1000
  max_replay_size: int = 1000000
  samples_per_insert: Optional[float] = 32.0
  sequence_length: int = 10
  sigma: float = 0.3
  replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE
  distillation_cost: Optional[float] = 1e-3
  entropy_regularizer_cost: Optional[float] = 1e-3


@dataclasses.dataclass
class SVG0Networks:
  """Structure containing the networks for SVG0."""

  policy_network: snt.Module
  critic_network: snt.Module
  prior_network: Optional[snt.Module]

  def __init__(
      self,
      policy_network: snt.Module,
      critic_network: snt.Module,
      prior_network: Optional[snt.Module] = None
  ):
    # This method is implemented (rather than added by the dataclass decorator)
    # in order to allow observation network to be passed as an arbitrary tensor
    # transformation rather than as a snt Module.
    # TODO(mwhoffman): use Protocol rather than Module/TensorTransformation.
    self.policy_network = policy_network
    self.critic_network = critic_network
    self.prior_network = prior_network

  def init(self, environment_spec: specs.EnvironmentSpec):
    """Initialize the networks given an environment spec."""
    # Get observation and action specs.
    act_spec = environment_spec.actions
    obs_spec = environment_spec.observations

    # Create variables for the policy and critic nets.
    _ = utils.create_variables(self.policy_network, [obs_spec])
    _ = utils.create_variables(self.critic_network, [obs_spec, act_spec])
    if self.prior_network is not None:
      _ = utils.create_variables(self.prior_network, [obs_spec])

  def make_policy(
      self,
  ) -> snt.Module:
    """Create a single network which evaluates the policy."""
    return self.policy_network

  def make_prior(
      self,
  ) -> snt.Module:
    """Create a single network which evaluates the prior."""
    behavior_prior = self.prior_network
    return behavior_prior


class SVG0Builder:
  """Builder for SVG0 which constructs individual components of the agent."""

  def __init__(self, config: SVG0Config):
    self._config = config

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
      sequence_length: int,
  ) -> List[reverb.Table]:
    """Create tables to insert data into."""
    if self._config.samples_per_insert is None:
      # We will take a samples_per_insert ratio of None to mean that there is
      # no limit, i.e. this only implies a min size limit.
      limiter = reverb.rate_limiters.MinSize(self._config.min_replay_size)

    else:
      error_buffer = max(1, self._config.samples_per_insert)
      limiter = reverb.rate_limiters.SampleToInsertRatio(
          min_size_to_sample=self._config.min_replay_size,
          samples_per_insert=self._config.samples_per_insert,
          error_buffer=error_buffer)

    extras_spec = {
        'log_prob': tf.ones(
            shape=(), dtype=tf.float32)
    }
    replay_table = reverb.Table(
        name=self._config.replay_table_name,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=self._config.max_replay_size,
        rate_limiter=limiter,
        signature=reverb_adders.SequenceAdder.signature(
            environment_spec,
            extras_spec=extras_spec,
            sequence_length=sequence_length + 1))

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

    # TODO(b/155086959): Fix type stubs and remove.
    return iter(dataset)  # pytype: disable=wrong-arg-types

  def make_adder(
      self,
      replay_client: reverb.Client,
  ) -> adders.Adder:
    """Create an adder which records data generated by the actor/environment."""
    return reverb_adders.SequenceAdder(
        client=replay_client,
        sequence_length=self._config.sequence_length+1,
        priority_fns={self._config.replay_table_name: lambda x: 1.},
        period=self._config.sequence_length,
        end_of_episode_behavior=reverb_adders.EndBehavior.CONTINUE,
        )

  def make_actor(
      self,
      policy_network: snt.Module,
      adder: Optional[adders.Adder] = None,
      variable_source: Optional[core.VariableSource] = None,
      deterministic_policy: Optional[bool] = False,
  ):
    """Create an actor instance."""
    if variable_source:
      # Create the variable client responsible for keeping the actor up-to-date.
      variable_client = variable_utils.VariableClient(
          client=variable_source,
          variables={'policy': policy_network.variables},
          update_period=1000,
      )

      # Make sure not to use a random policy after checkpoint restoration by
      # assigning variables before running the environment loop.
      variable_client.update_and_wait()

    else:
      variable_client = None

    # Create the actor which defines how we take actions.
    return acting.SVG0Actor(
        policy_network=policy_network,
        adder=adder,
        variable_client=variable_client,
        deterministic_policy=deterministic_policy
    )

  def make_learner(
      self,
      networks: Tuple[SVG0Networks, SVG0Networks],
      dataset: Iterator[reverb.ReplaySample],
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      checkpoint: bool = False,
  ):
    """Creates an instance of the learner."""
    online_networks, target_networks = networks

    # The learner updates the parameters (and initializes them).
    return learning.SVG0Learner(
        policy_network=online_networks.policy_network,
        critic_network=online_networks.critic_network,
        target_policy_network=target_networks.policy_network,
        target_critic_network=target_networks.critic_network,
        prior_network=online_networks.prior_network,
        target_prior_network=target_networks.prior_network,
        policy_optimizer=self._config.policy_optimizer,
        critic_optimizer=self._config.critic_optimizer,
        prior_optimizer=self._config.prior_optimizer,
        distillation_cost=self._config.distillation_cost,
        entropy_regularizer_cost=self._config.entropy_regularizer_cost,
        discount=self._config.discount,
        target_update_period=self._config.target_update_period,
        dataset_iterator=dataset,
        counter=counter,
        logger=logger,
        checkpoint=checkpoint,
    )


class SVG0(agent.Agent):
  """SVG0 Agent with prior.

  This implements a single-process SVG0 agent. This is an actor-critic algorithm
  that generates data via a behavior policy, inserts N-step transitions into
  a replay buffer, and periodically updates the policy (and as a result the
  behavior) by sampling uniformly from this buffer.
  """

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy_network: snt.Module,
      critic_network: snt.Module,
      discount: float = 0.99,
      batch_size: int = 256,
      prefetch_size: int = 4,
      target_update_period: int = 100,
      prior_network: Optional[snt.Module] = None,
      policy_optimizer: Optional[snt.Optimizer] = None,
      critic_optimizer: Optional[snt.Optimizer] = None,
      prior_optimizer: Optional[snt.Optimizer] = None,
      distillation_cost: Optional[float] = 1e-3,
      entropy_regularizer_cost: Optional[float] = 1e-3,
      min_replay_size: int = 1000,
      max_replay_size: int = 1000000,
      samples_per_insert: float = 32.0,
      sequence_length: int = 10,
      sigma: float = 0.3,
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
      discount: discount to use for TD updates.
      batch_size: batch size for updates.
      prefetch_size: size to prefetch from replay.
      target_update_period: number of learner steps to perform before updating
        the target networks.
      prior_network: an optional `behavior prior` to regularize against.
      policy_optimizer: optimizer for the policy network updates.
      critic_optimizer: optimizer for the critic network updates.
      prior_optimizer: optimizer for the prior network updates.
      distillation_cost: a multiplier to be used when adding distillation
        against the prior to the losses.
      entropy_regularizer_cost: a multiplier used for per state sample based
        entropy added to the actor loss.
      min_replay_size: minimum replay size before updating.
      max_replay_size: maximum replay size.
      samples_per_insert: number of samples to take from replay for every insert
        that is made.
      sequence_length: number of timesteps to store for each trajectory.
      sigma: standard deviation of zero-mean, Gaussian exploration noise.
      replay_table_name: string indicating what name to give the replay table.
      counter: counter object used to keep track of steps.
      logger: logger object to be used by learner.
      checkpoint: boolean indicating whether to checkpoint the learner.
    """
    # Create the Builder object which will internally create agent components.
    builder = SVG0Builder(
        # TODO(mwhoffman): pass the config dataclass in directly.
        # TODO(mwhoffman): use the limiter rather than the workaround below.
        # Right now this modifies min_replay_size and samples_per_insert so that
        # they are not controlled by a limiter and are instead handled by the
        # Agent base class (the above TODO directly references this behavior).
        SVG0Config(
            discount=discount,
            batch_size=batch_size,
            prefetch_size=prefetch_size,
            target_update_period=target_update_period,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            prior_optimizer=prior_optimizer,
            distillation_cost=distillation_cost,
            entropy_regularizer_cost=entropy_regularizer_cost,
            min_replay_size=1,  # Let the Agent class handle this.
            max_replay_size=max_replay_size,
            samples_per_insert=None,  # Let the Agent class handle this.
            sequence_length=sequence_length,
            sigma=sigma,
            replay_table_name=replay_table_name,
        ))

    # TODO(mwhoffman): pass the network dataclass in directly.
    online_networks = SVG0Networks(policy_network=policy_network,
                                   critic_network=critic_network,
                                   prior_network=prior_network,)

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
    policy_network = online_networks.make_policy()

    # Create the replay server and grab its address.
    replay_tables = builder.make_replay_tables(environment_spec,
                                               sequence_length)
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
