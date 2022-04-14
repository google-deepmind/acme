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

from typing import Any, Callable, Dict, Optional, Union, Sequence

from acme import core
from acme.jax import networks as networks_lib
from acme.jax import savers
from acme.jax import types
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from acme.utils import lp_utils
import jax
import launchpad as lp

AgentNetwork = Any
NetworkFactory = Callable[[], AgentNetwork]
# It will be treated as Dict[str, Any]. Proper support is tracked b/109648354.
NestedLogger = Union[loggers.Logger, Dict[str, 'NestedLogger']]  # pytype: disable=not-supported-yet
LearnerFactory = Callable[[
    types.PRNGKey,
    AgentNetwork,
    Optional[counting.Counter],
    Optional[NestedLogger],
], core.Learner]
EvaluatorFactory = Callable[
    [types.PRNGKey, core.VariableSource, counting.Counter], core.Worker]


class OfflineDistributedLayout:
  """Program definition for an offline distributed agent based on a builder.

  It is distributed in the sense that evaluators run on different machines than
  learner.
  """

  def __init__(
      self,
      seed: int,
      network_factory: NetworkFactory,
      make_learner: LearnerFactory,
      evaluator_factories: Sequence[EvaluatorFactory] = (),
      save_logs: bool = False,
      log_every: float = 10.0,
      max_number_of_steps: Optional[int] = None,
      workdir: str = '~/acme',
  ):

    self._seed = seed
    self._make_learner = make_learner
    self._evaluator_factories = evaluator_factories
    self._network_factory = network_factory
    self._save_logs = save_logs
    self._log_every = log_every
    self._max_number_of_steps = max_number_of_steps
    self._workdir = workdir

  def counter(self):
    kwargs = {'directory': self._workdir, 'add_uid': self._workdir == '~/acme'}
    return savers.CheckpointingRunner(
        counting.Counter(), subdirectory='counter', time_delta_minutes=5,
        **kwargs)

  def learner(
      self,
      random_key: networks_lib.PRNGKey,
      counter: counting.Counter,
  ):
    """The Learning part of the agent."""
    # Counter and logger.
    counter = counting.Counter(counter, 'learner')
    logger = loggers.make_default_logger(
        'learner', self._save_logs, time_delta=self._log_every,
        asynchronous=True, serialize_fn=utils.fetch_devicearray,
        steps_key='learner_steps')

    # Create the learner.
    networks = self._network_factory()
    learner = self._make_learner(random_key, networks, counter, logger)

    kwargs = {'directory': self._workdir, 'add_uid': self._workdir == '~/acme'}
    # Return the learning agent.
    return savers.CheckpointingRunner(
        learner, subdirectory='learner', time_delta_minutes=5, **kwargs)

  def coordinator(self, counter: counting.Counter, max_learner_steps: int):
    return lp_utils.StepsLimiter(counter, max_steps=max_learner_steps,
                                 steps_key='learner_steps')

  def build(self, name='agent'):
    """Build the distributed agent topology."""
    program = lp.Program(name=name)

    key = jax.random.PRNGKey(self._seed)

    with program.group('counter'):
      counter = program.add_node(lp.CourierNode(self.counter))
      if self._max_number_of_steps is not None:
        _ = program.add_node(
            lp.CourierNode(self.coordinator, counter,
                           self._max_number_of_steps))

    learner_key, key = jax.random.split(key)
    with program.group('learner'):
      learner = program.add_node(
          lp.CourierNode(self.learner, learner_key, counter))

    with program.group('evaluator'):
      for evaluator in self._evaluator_factories:
        evaluator_key, key = jax.random.split(key)
        program.add_node(
            lp.CourierNode(evaluator, evaluator_key, learner, counter))

    return program
