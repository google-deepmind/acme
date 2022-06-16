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
import sys
from typing import Any, Callable, Optional, Sequence

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
MakeActorFn = Callable[
    [types.PRNGKey, PolicyNetwork, specs.EnvironmentSpec, core.VariableSource],
    core.Actor]
NetworkFactory = Callable[[specs.EnvironmentSpec], AgentNetwork]
PolicyFactory = Callable[[AgentNetwork], PolicyNetwork]
EvaluatorFactory = Callable[[
    types.PRNGKey,
    core.VariableSource,
    counting.Counter,
    MakeActorFn,
], core.Worker]


@dataclasses.dataclass
class Config:
  """Config which defines aspects of constructing an experiment.

  Attributes:
    builder: Builds components of an RL agent (Learner, Actor...).
    network_factory: Builds networks used by the agent.
    policy_network_factory: Policy network factory which is used actors to
      perform inference.
    environment_factory: Returns an instance of an environment.
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
    seed: Seed used for agent initialization.
    max_number_of_steps: How many environment steps to perform. Infinite by
      default.
    logger_factory: Loggers factory used to construct loggers for learner,
      actors and evaluators.
  """
  # Below fields must be explicitly specified for any Agent.
  builder: builders.ActorLearnerBuilder
  network_factory: NetworkFactory
  policy_network_factory: PolicyFactory
  environment_factory: types.EnvironmentFactory
  # Fields below are optional. If you just started with Acme do not worry about
  # them. You might need them later when you want to customize your RL agent.
  # TODO(stanczyk): Introduce a marker for the default value (instead of None).
  evaluator_factories: Optional[Sequence[EvaluatorFactory]] = None
  # TODO(mwhoffman): Change the way network_factory, policy_network_factory
  # and eval_policy_network_factory are specified.
  eval_policy_network_factory: Optional[PolicyFactory] = None
  environment_spec: Optional[specs.EnvironmentSpec] = None
  observers: Sequence[observers_lib.EnvLoopObserver] = ()
  seed: int = 0
  # TODO(stanczyk): Make this field required.
  max_number_of_steps: int = sys.maxsize
  logger_factory: loggers.LoggerFactory = experiment_utils.make_experiment_logger

  # TODO(stanczyk): Make get_evaluator_factories a standalone function.
  def get_evaluator_factories(self):
    if self.evaluator_factories is not None:
      return self.evaluator_factories
    assert self.eval_policy_network_factory is not None, (
        'You need to provide `eval_policy_network_factory` to Config'
        ' when `evaluator_factories` are not specified. To disable evaluation '
        'altogether just set `evaluator_factories = []`')
    return [
        default_evaluator_factory(
            environment_factory=self.environment_factory,
            network_factory=self.network_factory,
            policy_factory=self.eval_policy_network_factory,
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

    actor = make_actor(actor_key, policy_factory(networks), environment_spec,
                       variable_source)

    # Create logger and counter.
    counter = counting.Counter(counter, 'evaluator')
    logger = logger_factory('evaluator', 'actor_steps', 0)

    # Create the run loop and return it.
    return environment_loop.EnvironmentLoop(
        environment, actor, counter, logger, observers=observers)

  return evaluator
