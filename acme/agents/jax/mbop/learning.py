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

"""Combined offline learning of world model, policy and N-step return."""

import dataclasses
import functools
import itertools
import time
from typing import Any, Callable, Iterator, List, Optional

from acme import core
from acme import types
from acme.agents.jax import bc
from acme.agents.jax.mbop import ensemble
from acme.agents.jax.mbop import losses as mbop_losses
from acme.agents.jax.mbop import networks as mbop_networks
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import optax


@dataclasses.dataclass
class TrainingState:
  """States of the world model, policy prior and n-step return learners."""
  world_model: Any
  policy_prior: Any
  n_step_return: Any


LoggerFn = Callable[[str, str], loggers.Logger]

# Creates a world model learner.
MakeWorldModelLearner = Callable[[
    LoggerFn,
    counting.Counter,
    jax_types.PRNGKey,
    Iterator[types.Transition],
    mbop_networks.WorldModelNetwork,
    mbop_losses.TransitionLoss,
], core.Learner]

# Creates a policy prior learner.
MakePolicyPriorLearner = Callable[[
    LoggerFn,
    counting.Counter,
    jax_types.PRNGKey,
    Iterator[types.Transition],
    mbop_networks.PolicyPriorNetwork,
    mbop_losses.TransitionLoss,
], core.Learner]

# Creates an n-step return model learner.
MakeNStepReturnLearner = Callable[[
    LoggerFn,
    counting.Counter,
    jax_types.PRNGKey,
    Iterator[types.Transition],
    mbop_networks.NStepReturnNetwork,
    mbop_losses.TransitionLoss,
], core.Learner]


def make_ensemble_regressor_learner(
    name: str,
    num_networks: int,
    logger_fn: loggers.LoggerFactory,
    counter: counting.Counter,
    rng_key: jnp.ndarray,
    iterator: Iterator[types.Transition],
    base_network: networks_lib.FeedForwardNetwork,
    loss: mbop_losses.TransitionLoss,
    optimizer: optax.GradientTransformation,
    num_sgd_steps_per_step: int,
):
  """Creates an ensemble regressor learner from the base network.

  Args:
    name: Name of the learner used for logging and counters.
    num_networks: Number of networks in the ensemble.
    logger_fn: Constructs a logger for a label.
    counter: Parent counter object.
    rng_key: Random key.
    iterator: An iterator of time-batched transitions used to train the
      networks.
    base_network: Base network for the ensemble.
    loss: Training loss to use.
    optimizer: Optax optimizer.
    num_sgd_steps_per_step: Number of gradient updates per step.

  Returns:
    An ensemble regressor learner.
  """
  mbop_ensemble = ensemble.make_ensemble(base_network, ensemble.apply_all,
                                         num_networks)
  local_counter = counting.Counter(parent=counter, prefix=name)
  local_logger = logger_fn(name,
                           local_counter.get_steps_key()) if logger_fn else None

  def loss_fn(apply_fn: Callable[..., networks_lib.NetworkOutput],
              params: networks_lib.Params, key: jnp.ndarray,
              transitions: types.Transition) -> jnp.ndarray:
    del key
    return loss(functools.partial(apply_fn, params), transitions)

  # This is effectively a regressor learner.
  return bc.BCLearner(
      mbop_ensemble,
      rng_key,
      loss_fn,
      optimizer,
      iterator,
      num_sgd_steps_per_step,
      logger=local_logger,
      counter=local_counter)


class MBOPLearner(core.Learner):
  """Model-Based Offline Planning (MBOP) learner.

  See https://arxiv.org/abs/2008.05556 for more information.
  """

  def __init__(self,
               networks: mbop_networks.MBOPNetworks,
               losses: mbop_losses.MBOPLosses,
               iterator: Iterator[types.Transition],
               rng_key: jax_types.PRNGKey,
               logger_fn: LoggerFn,
               make_world_model_learner: MakeWorldModelLearner,
               make_policy_prior_learner: MakePolicyPriorLearner,
               make_n_step_return_learner: MakeNStepReturnLearner,
               counter: Optional[counting.Counter] = None):
    """Creates an MBOP learner.

    Args:
      networks: One network per model.
      losses: One loss per model.
      iterator: An iterator of time-batched transitions used to train the
        networks.
      rng_key: Random key.
      logger_fn: Constructs a logger for a label.
      make_world_model_learner: Function to create the world model learner.
      make_policy_prior_learner: Function to create the policy prior learner.
      make_n_step_return_learner: Function to create the n-step return learner.
      counter: Parent counter object.
    """
    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger_fn('', 'steps')

    # Prepare iterators for the learners, to not split the data (preserve sample
    # efficiency).
    world_model_iterator, policy_prior_iterator, n_step_return_iterator = (
        itertools.tee(iterator, 3))

    world_model_key, policy_prior_key, n_step_return_key = jax.random.split(
        rng_key, 3)

    self._world_model = make_world_model_learner(logger_fn, self._counter,
                                                 world_model_key,
                                                 world_model_iterator,
                                                 networks.world_model_network,
                                                 losses.world_model_loss)
    self._policy_prior = make_policy_prior_learner(
        logger_fn, self._counter, policy_prior_key, policy_prior_iterator,
        networks.policy_prior_network, losses.policy_prior_loss)
    self._n_step_return = make_n_step_return_learner(
        logger_fn, self._counter, n_step_return_key, n_step_return_iterator,
        networks.n_step_return_network, losses.n_step_return_loss)
    # Start recording timestamps after the first learning step to not report
    # "warmup" time.
    self._timestamp = None
    self._learners = {
        'world_model': self._world_model,
        'policy_prior': self._policy_prior,
        'n_step_return': self._n_step_return
    }

  def step(self):
    # Step the world model, policy learner and n-step return learners.
    self._world_model.step()
    self._policy_prior.step()
    self._n_step_return.step()

    # Compute the elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp
    # Increment counts and record the current time.
    counts = self._counter.increment(steps=1, walltime=elapsed_time)
    # Attempt to write the logs.
    self._logger.write({**counts})

  def get_variables(self, names: List[str]) -> List[types.NestedArray]:
    variables = []
    for name in names:
      # Variables will be prefixed by the learner names. If separator is not
      # found, learner_name=name, which is OK.
      learner_name, _, variable_name = name.partition('-')
      learner = self._learners[learner_name]
      variables.extend(learner.get_variables([variable_name]))
    return variables

  def save(self) -> TrainingState:
    return TrainingState(
        world_model=self._world_model.save(),
        policy_prior=self._policy_prior.save(),
        n_step_return=self._n_step_return.save())

  def restore(self, state: TrainingState):
    self._world_model.restore(state.world_model)
    self._policy_prior.restore(state.policy_prior)
    self._n_step_return.restore(state.n_step_return)
