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
import jax

AgentNetwork = Any
PolicyNetwork = Any
MakeActorFn = Callable[[types.PRNGKey, PolicyNetwork, core.VariableSource],
                       core.Actor]
NetworkFactory = Callable[[specs.EnvironmentSpec], AgentNetwork]
PolicyFactory = Callable[[AgentNetwork], PolicyNetwork]
EvaluatorFactory = Callable[[
    types.PRNGKey,
    core.VariableSource,
    counting.Counter,
    MakeActorFn,
], core.Worker]
LoggerLabel = str
LoggerStepsKey = str
LoggerFn = Callable[[LoggerLabel, LoggerStepsKey], loggers.Logger]


@dataclasses.dataclass
class Config:
  """Experiment configuration which defines all aspects of constructing an Agent.

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
    save_logs: Whether to save logs to disk or external services (depends on
      specific runtime used).
  """
  # Below fields must be explicitly specified for any Agent.
  builder: builders.GenericActorLearnerBuilder
  network_factory: NetworkFactory
  policy_network_factory: PolicyFactory
  environment_factory: types.EnvironmentFactory
  # Fields below are optional. If you just started with Acme do not worry about
  # them. You might need them later when you want to customize your RL agent.
  evaluator_factories: Optional[Sequence[EvaluatorFactory]] = None
  eval_policy_network_factory: Optional[PolicyFactory] = None
  environment_spec: Optional[specs.EnvironmentSpec] = None
  observers: Sequence[observers_lib.EnvLoopObserver] = ()
  seed: int = 0
  max_number_of_steps: int = sys.maxsize
  save_logs: bool = True

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
            log_to_bigtable=self.save_logs)
    ]


def default_evaluator_factory(
    environment_factory: types.EnvironmentFactory,
    network_factory: NetworkFactory,
    policy_factory: PolicyFactory,
    observers: Sequence[observers_lib.EnvLoopObserver] = (),
    log_to_bigtable: bool = False,
    logger_fn: Optional[LoggerFn] = None) -> EvaluatorFactory:
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
    return environment_loop.EnvironmentLoop(
        environment, actor, counter, logger, observers=observers)

  return evaluator
