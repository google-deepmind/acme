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

from typing import Callable, Dict, Optional, Sequence

from acme import core
from acme import environment_loop
from acme import specs
from acme.agents.jax import builders
from acme.jax import experiments
from acme.jax import types
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from acme.utils import observers as observers_lib
import jax
import launchpad as lp

# TODO(stanczyk): Remove when use cases are ported to the new location.
EvaluatorFactory = experiments.config.EvaluatorFactory
AgentNetwork = experiments.config.AgentNetwork
PolicyNetwork = experiments.config.PolicyNetwork
NetworkFactory = experiments.config.NetworkFactory
PolicyFactory = experiments.config.DeprecatedPolicyFactory
MakeActorFn = experiments.config.MakeActorFn
LoggerLabel = loggers.LoggerLabel
LoggerStepsKey = loggers.LoggerStepsKey
LoggerFn = Callable[[LoggerLabel, LoggerStepsKey], loggers.Logger]
EvaluatorFactory = experiments.config.EvaluatorFactory

ActorId = int

SnapshotModelFactory = Callable[
    [experiments.config.AgentNetwork, specs.EnvironmentSpec],
    Dict[str, Callable[[core.VariableSource], types.ModelToSnapshot]]]

CheckpointingConfig = experiments.CheckpointingConfig


def default_evaluator_factory(
    environment_factory: types.EnvironmentFactory,
    network_factory: NetworkFactory,
    policy_factory: PolicyFactory,
    observers: Sequence[observers_lib.EnvLoopObserver] = (),
    save_logs: bool = False,
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
    environment_spec = specs.make_environment_spec(environment)
    policy = policy_factory(network_factory(environment_spec))

    actor = make_actor(actor_key, policy, environment_spec, variable_source)

    # Create logger and counter.
    counter = counting.Counter(counter, 'evaluator')
    if logger_fn is not None:
      logger = logger_fn('evaluator', 'actor_steps')
    else:
      logger = loggers.make_default_logger(
          'evaluator', save_logs, steps_key='actor_steps')

    # Create the run loop and return it.
    return environment_loop.EnvironmentLoop(
        environment, actor, counter, logger, observers=observers)

  return evaluator


def get_default_logger_fn(
    save_logs: bool = False,
    log_every: float = 10) -> Callable[[ActorId], loggers.Logger]:
  """Creates an actor logger."""

  def create_logger(actor_id: ActorId):
    return loggers.make_default_logger(
        'actor',
        save_data=(save_logs and actor_id == 0),
        time_delta=log_every,
        steps_key='actor_steps')

  return create_logger


def logger_factory(
    learner_logger_fn: Optional[Callable[[], loggers.Logger]] = None,
    actor_logger_fn: Optional[Callable[[ActorId], loggers.Logger]] = None,
    save_logs: bool = True,
    log_every: float = 10.0) -> Callable[[str, str, int], loggers.Logger]:
  """Builds a logger factory used by the experiments.config."""

  def factory(label: str,
              steps_key: Optional[str] = None,
              task_id: Optional[int] = None):
    if task_id is None:
      task_id = 0
    if steps_key is None:
      steps_key = f'{label}_steps'
    if label == 'learner' and learner_logger_fn:
      return learner_logger_fn()
    if label == 'actor':
      if actor_logger_fn:
        return actor_logger_fn(task_id)
      else:
        return get_default_logger_fn(save_logs)(task_id)
    if label == 'evaluator':
      return loggers.make_default_logger(
          label, save_logs, time_delta=log_every, steps_key=steps_key)
    return None

  return factory


class DistributedLayout:
  """Program definition for a distributed agent based on a builder.

  DEPRECATED: Use make_distributed_experiment directly.
  """

  def __init__(
      self,
      seed: int,
      environment_factory: types.EnvironmentFactory,
      network_factory: experiments.config.NetworkFactory,
      builder: builders.ActorLearnerBuilder,
      policy_network: experiments.config.DeprecatedPolicyFactory,
      num_actors: int,
      environment_spec: Optional[specs.EnvironmentSpec] = None,
      learner_logger_fn: Optional[Callable[[], loggers.Logger]] = None,
      actor_logger_fn: Optional[Callable[[ActorId], loggers.Logger]] = None,
      evaluator_factories: Sequence[experiments.config.EvaluatorFactory] = (),
      prefetch_size: int = 1,
      save_logs: bool = False,
      max_number_of_steps: Optional[int] = None,
      observers: Sequence[observers_lib.EnvLoopObserver] = (),
      multithreading_colocate_learner_and_reverb: bool = False,
      checkpointing_config: Optional[CheckpointingConfig] = None,
      make_snapshot_models: Optional[SnapshotModelFactory] = None):
    del prefetch_size
    self._experiment_config = experiments.config.ExperimentConfig(
        builder=builder,
        environment_factory=environment_factory,
        environment_spec=environment_spec,
        network_factory=network_factory,
        policy_network_factory=policy_network,
        evaluator_factories=evaluator_factories,
        observers=observers,
        seed=seed,
        max_num_actor_steps=max_number_of_steps,
        logger_factory=logger_factory(learner_logger_fn, actor_logger_fn,
                                      save_logs))
    self._num_actors = num_actors
    self._multithreading_colocate_learner_and_reverb = (
        multithreading_colocate_learner_and_reverb)
    self._checkpointing_config = checkpointing_config
    self._make_snapshot_models = make_snapshot_models

  def build(self, name='agent', program: Optional[lp.Program] = None):
    """Build the distributed agent topology."""

    return experiments.make_distributed_experiment(
        self._experiment_config,
        self._num_actors,
        multithreading_colocate_learner_and_reverb=self
        ._multithreading_colocate_learner_and_reverb,
        checkpointing_config=self._checkpointing_config,
        make_snapshot_models=self._make_snapshot_models,
        name=name,
        program=program)
