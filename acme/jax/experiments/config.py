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

"""JAX experiment config."""

import dataclasses
from typing import Any, Callable, Generic, Iterator, Optional, Sequence

from acme import core
from acme import environment_loop
from acme import specs
from acme.agents.jax import builders
from acme.jax import types
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from acme.utils import observers as observers_lib
from acme.utils import experiment_utils
import jax

AgentNetwork = Any
PolicyNetwork = Any
EvaluationFlag = bool
MakeActorFn = Callable[
    [types.PRNGKey, PolicyNetwork, specs.EnvironmentSpec, core.VariableSource],
    core.Actor]
NetworkFactory = Callable[[specs.EnvironmentSpec], AgentNetwork]
DeprecatedPolicyFactory = Callable[[AgentNetwork], PolicyNetwork]
PolicyFactory = Callable[[AgentNetwork, specs.EnvironmentSpec, EvaluationFlag],
                         PolicyNetwork]
EvaluatorFactory = Callable[[
    types.PRNGKey,
    core.VariableSource,
    counting.Counter,
    MakeActorFn,
], core.Worker]


@dataclasses.dataclass
class ExperimentConfig:
  """Config which defines aspects of constructing an experiment.

  Attributes:
    builder: Builds components of an RL agent (Learner, Actor...).
    network_factory: Builds networks used by the agent.
    environment_factory: Returns an instance of an environment.
    max_num_actor_steps: How many environment steps to perform.
    seed: Seed used for agent initialization.
    policy_network_factory: Policy network factory which is used actors to
      perform inference.
    evaluator_factories: Factories of policy evaluators. When not specified the
      default evaluators are constructed using eval_policy_network_factory. Set
      to an empty list to disable evaluators.
    eval_policy_network_factory: Policy network factory used by evaluators.
      Should be specified to use the default evaluators (when
      evaluator_factories is not provided).
    environment_spec: Specification of the environment. Can be specified to
      reduce the number of times environment_factory is invoked (for performance
      or resource usage reasons).
    observers: Observers used for extending logs with custom information.
    logger_factory: Loggers factory used to construct loggers for learner,
      actors and evaluators.
  """
  # Below fields must be explicitly specified for any Agent.
  builder: builders.ActorLearnerBuilder
  network_factory: NetworkFactory
  environment_factory: types.EnvironmentFactory
  max_num_actor_steps: int
  seed: int
  # policy_network_factory is deprecated. Use builder.make_policy to
  # create the policy.
  policy_network_factory: Optional[DeprecatedPolicyFactory] = None
  # Fields below are optional. If you just started with Acme do not worry about
  # them. You might need them later when you want to customize your RL agent.
  # TODO(stanczyk): Introduce a marker for the default value (instead of None).
  evaluator_factories: Optional[Sequence[EvaluatorFactory]] = None
  # eval_policy_network_factory is deprecated. Use builder.make_policy to
  # create the policy.
  eval_policy_network_factory: Optional[DeprecatedPolicyFactory] = None
  environment_spec: Optional[specs.EnvironmentSpec] = None
  observers: Sequence[observers_lib.EnvLoopObserver] = ()
  logger_factory: loggers.LoggerFactory = experiment_utils.make_experiment_logger

  # TODO(stanczyk): Make get_evaluator_factories a standalone function.
  def get_evaluator_factories(self):
    """Constructs the evaluator factories."""
    if self.evaluator_factories is not None:
      return self.evaluator_factories

    def eval_policy_factory(networks: AgentNetwork,
                            environment_spec: specs.EnvironmentSpec,
                            evaluation: EvaluationFlag) -> PolicyNetwork:
      del evaluation
      # The config factory has precedence until all agents are migrated to use
      # builder.make_policy
      if self.eval_policy_network_factory is not None:
        return self.eval_policy_network_factory(networks)
      else:
        return self.builder.make_policy(
            networks=networks,
            environment_spec=environment_spec,
            evaluation=True)

    return [
        default_evaluator_factory(
            environment_factory=self.environment_factory,
            network_factory=self.network_factory,
            policy_factory=eval_policy_factory,
            logger_factory=self.logger_factory,
            observers=self.observers)
    ]


@dataclasses.dataclass
class OfflineExperimentConfig(Generic[builders.Networks, builders.Policy,
                                      builders.Sample]):
  """Config which defines aspects of constructing an offline RL experiment.

  This class is similar to the ExperimentConfig, but is tailored to offline RL
  setting, so it excludes attributes related to training via interaction with
  the environment (max_num_actor_steps, policy_network_factory) and instead
  includes attributes specific to learning from demonstration.

  Attributes:
    builder: Builds components of an offline RL agent (Learner and Evaluator).
    network_factory: Builds networks used by the agent.
    demonstration_dataset_factory: Function that returns an iterator over
      demonstrations.
    environment_spec: Specification of the environment.
    max_num_learner_steps: How many learner steps to perform.
    seed: Seed used for agent initialization.
    evaluator_factories: Factories of policy evaluators. When not specified the
      default evaluators are constructed using eval_policy_network_factory. Set
      to an empty list to disable evaluators.
    eval_policy_factory: Policy factory used by evaluators. Should be specified
      to use the default evaluators (when evaluator_factories is not provided).
    environment_factory: Returns an instance of an environment to be used for
      evaluation. Should be specified to use the default evaluators (when
      evaluator_factories is not provided).
    observers: Observers used for extending logs with custom information.
    logger_factory: Loggers factory used to construct loggers for learner,
      actors and evaluators.
  """
  # Below fields must be explicitly specified for any Agent.
  builder: builders.OfflineBuilder[builders.Networks, builders.Policy,
                                   builders.Sample]
  network_factory: Callable[[specs.EnvironmentSpec], builders.Networks]
  demonstration_dataset_factory: Callable[[types.PRNGKey],
                                          Iterator[builders.Sample]]
  environment_factory: types.EnvironmentFactory
  max_num_learner_steps: int
  seed: int
  # Fields below are optional. If you just started with Acme do not worry about
  # them. You might need them later when you want to customize your RL agent.
  # TODO(stanczyk): Introduce a marker for the default value (instead of None).
  evaluator_factories: Optional[Sequence[EvaluatorFactory]] = None
  environment_spec: Optional[specs.EnvironmentSpec] = None
  observers: Sequence[observers_lib.EnvLoopObserver] = ()
  logger_factory: loggers.LoggerFactory = experiment_utils.make_experiment_logger

  # TODO(stanczyk): Make get_evaluator_factories a standalone function.
  def get_evaluator_factories(self):
    """Constructs the evaluator factories."""
    if self.evaluator_factories is not None:
      return self.evaluator_factories
    if self.environment_factory is None:
      raise ValueError(
          'You need to set `environment_factory` in `OfflineExperimentConfig` '
          'when `evaluator_factories` are not specified. To disable evaluation '
          'altogether just set `evaluator_factories = []`')

    return [
        default_evaluator_factory(
            environment_factory=self.environment_factory,
            network_factory=self.network_factory,
            policy_factory=self.builder.make_policy,
            logger_factory=self.logger_factory,
            observers=self.observers)
    ]


def default_evaluator_factory(
    environment_factory: types.EnvironmentFactory,
    network_factory: NetworkFactory,
    policy_factory: PolicyFactory,
    logger_factory: loggers.LoggerFactory,
    observers: Sequence[observers_lib.EnvLoopObserver] = (),
) -> EvaluatorFactory:
  """Returns a default evaluator process."""

  def evaluator(
      random_key: types.PRNGKey,
      variable_source: core.VariableSource,
      counter: counting.Counter,
      make_actor: MakeActorFn,
  ):
    """The evaluation process."""

    # Create environment and evaluator networks
    environment_key, actor_key = jax.random.split(random_key)
    # Environments normally require uint32 as a seed.
    environment = environment_factory(utils.sample_uint32(environment_key))
    environment_spec = specs.make_environment_spec(environment)
    networks = network_factory(environment_spec)
    policy = policy_factory(networks, environment_spec, True)
    actor = make_actor(actor_key, policy, environment_spec, variable_source)

    # Create logger and counter.
    counter = counting.Counter(counter, 'evaluator')
    logger = logger_factory('evaluator', 'actor_steps', 0)

    # Create the run loop and return it.
    return environment_loop.EnvironmentLoop(
        environment, actor, counter, logger, observers=observers)

  return evaluator


def make_policy(experiment: ExperimentConfig, networks: AgentNetwork,
                environment_spec: specs.EnvironmentSpec,
                evaluation: bool) -> PolicyNetwork:
  """Constructs a policy. It is only meant to be used internally."""
  # TODO(sabela): remove and update callers once all agents use
  # builder.make_policy
  if not evaluation and experiment.policy_network_factory:
    return experiment.policy_network_factory(networks)
  if evaluation and experiment.eval_policy_network_factory:
    return experiment.eval_policy_network_factory(networks)
  return experiment.builder.make_policy(
      networks=networks,
      environment_spec=environment_spec,
      evaluation=evaluation)


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
      replay duties, however this does pose a risk of OOM since items that would
      otherwise be removed are temporarily kept alive for checkpointing
      purposes.
      Note: Since replay buffers tend to be quite large O(100GiB), writing can
        take up to 10 minutes so keep that in mind when setting this frequency.
  """
  max_to_keep: int = 1
  directory: str = '~/acme'
  add_uid: bool = True
  replay_checkpointing_time_delta_minutes: Optional[int] = None
