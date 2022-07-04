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

"""RL agent Builder interface."""

import abc
from typing import Generic, Iterator, List, Optional, TypeVar

from acme import adders
from acme import core
from acme import specs
from acme.jax import networks as networks_lib
from acme.utils import counting
from acme.utils import loggers
import reverb

Networks = TypeVar('Networks')  # Container for all agent network components.
Policy = TypeVar('Policy')  # Function or container for agent policy functions.
Sample = TypeVar('Sample')  # Sample from the demonstrations or replay buffer.


class OfflineBuilder(abc.ABC, Generic[Networks, Policy, Sample]):
  """Interface for defining the components of an offline RL agent.

  Implementations of this interface contain a complete specification of a
  concrete offline RL agent. An instance of this class can be used to build an
  offline RL agent that operates either locally or in a distributed setup.
  """

  @abc.abstractmethod
  def make_learner(
      self,
      *,
      random_key: networks_lib.PRNGKey,
      networks: Networks,
      dataset: Iterator[Sample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    """Creates an instance of the learner.

    Args:
      random_key: A key for random number generation.
      networks: struct describing the networks needed by the learner; this is
        specific to the learner in question.
      dataset: iterator over demonstration samples.
      logger_fn: factory providing loggers used for logging progress.
      environment_spec: A container for all relevant environment specs.
      counter: a Counter which allows for recording of counts (learner steps,
        evaluator steps, etc.) distributed throughout the agent.
    """

  @abc.abstractmethod
  def make_actor(
      self,
      *,
      random_key: networks_lib.PRNGKey,
      policy: Policy,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
  ) -> core.Actor:
    """Create an actor instance to be used for evaluation.

    Args:
      random_key: A key for random number generation.
      policy: Instance of a policy expected by the algorithm corresponding to
        this builder.
      environment_spec: A container for all relevant environment specs.
      variable_source: A source providing the necessary actor parameters.
    """

  @abc.abstractmethod
  def make_policy(self, networks: Networks,
                  environment_spec: specs.EnvironmentSpec,
                  evaluation: bool) -> Policy:
    """Creates the agent policy to be used for evaluation.

    Args:
      networks: struct describing the networks needed to generate the policy.
      environment_spec: struct describing the specs of the environment.
      evaluation: This flag is present for consistency with the
        ActorLearnerBuilder, in which case data-generating actors and evaluation
        actors can behave differently. For OfflineBuilders, this should be set
        to True.

    Returns:
      Policy to be used for evaluation. The exact form of this object may differ
      from one agent to the next; it could be a simple callable, a nest of
      callables, or an ActorCore for instance.
    """


class ActorLearnerBuilder(OfflineBuilder[Networks, Policy, Sample],
                          Generic[Networks, Policy, Sample]):
  """Defines an interface for defining the components of an RL agent.

  Implementations of this interface contain a complete specification of a
  concrete RL agent. An instance of this class can be used to build an
  RL agent which interacts with the environment either locally or in a
  distributed setup.
  """

  @abc.abstractmethod
  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy: Policy,
  ) -> List[reverb.Table]:
    """Create tables to insert data into.

    Args:
      environment_spec: A container for all relevant environment specs.
      policy: Agent's policy which can be used to extract the extras_spec.

    Returns:
      The replay tables used to store the experience the agent uses to train.
    """

  @abc.abstractmethod
  def make_dataset_iterator(
      self,
      replay_client: reverb.Client,
  ) -> Iterator[Sample]:
    """Create a dataset iterator to use for learning/updating the agent."""

  @abc.abstractmethod
  def make_adder(
      self,
      replay_client: reverb.Client,
  ) -> Optional[adders.Adder]:
    """Create an adder which records data generated by the actor/environment.

    Args:
      replay_client: Reverb Client which points to the replay server.
    """

  @abc.abstractmethod
  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy: Policy,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
      adder: Optional[adders.Adder] = None,
  ) -> core.Actor:
    """Create an actor instance.

    Args:
      random_key: A key for random number generation.
      policy: Instance of a policy expected by the algorithm corresponding to
        this builder.
      environment_spec: A container for all relevant environment specs.
      variable_source: A source providing the necessary actor parameters.
      adder: How data is recorded (e.g. added to replay).
    """

  @abc.abstractmethod
  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: Networks,
      dataset: Iterator[Sample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    """Creates an instance of the learner.

    Args:
      random_key: A key for random number generation.
      networks: struct describing the networks needed by the learner; this can
        be specific to the learner in question.
      dataset: iterator over samples from replay.
      logger_fn: factory providing loggers used for logging progress.
      environment_spec: A container for all relevant environment specs.
      replay_client: client which allows communication with replay. Note that
        this is only intended to be used for updating priorities. Samples should
        be obtained from `dataset`.
      counter: a Counter which allows for recording of counts (learner steps,
        actor steps, etc.) distributed throughout the agent.
    """

  def make_policy(self,
                  networks: Networks,
                  environment_spec: specs.EnvironmentSpec,
                  evaluation: bool = False) -> Policy:
    """Creates the agent policy.

       Creates the agent policy given the collection of network components and
       environment spec. An optional boolean can be given to indicate if the
       policy will be used for evaluation.

    Args:
      networks: struct describing the networks needed to generate the policy.
      environment_spec: struct describing the specs of the environment.
      evaluation: when true, a version of the policy to use for evaluation
        should be returned. This is algorithm-specific so if an algorithm makes
        no distinction between behavior and evaluation policies this boolean may
        be ignored.

    Returns:
      Behavior policy or evaluation policy for the agent.
    """
    # TODO(sabela): make abstract once all agents implement it.
    del networks, environment_spec, evaluation
    raise NotImplementedError

# TODO(sinopalnikov): deprecated, migrate all users and remove.
GenericActorLearnerBuilder = ActorLearnerBuilder
