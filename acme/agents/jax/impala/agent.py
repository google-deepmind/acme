# python3
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

"""Importance weighted advantage actor-critic (IMPALA) agent implementation."""

from typing import Callable

import acme
from acme import specs
from acme.agents import replay
from acme.agents.jax.impala import acting
from acme.agents.jax.impala import learning
from acme.agents.jax.impala import types
from acme.jax import networks
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import dm_env
import haiku as hk
import jax
import numpy as np
import optax


class IMPALA(acme.Actor):
  """IMPALA Agent."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      forward_fn: networks.PolicyValueRNN,
      unroll_fn: networks.PolicyValueRNN,
      initial_state_fn: Callable[[], hk.LSTMState],
      sequence_length: int,
      sequence_period: int,
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
      discount: float = 0.99,
      max_queue_size: int = 100000,
      batch_size: int = 16,
      learning_rate: float = 1e-3,
      entropy_cost: float = 0.01,
      baseline_cost: float = 0.5,
      seed: int = 0,
      max_abs_reward: float = np.inf,
      max_gradient_norm: float = np.inf,
  ):

    # Data is handled by the reverb replay queue.
    num_actions = environment_spec.actions.num_values
    self._logger = logger or loggers.TerminalLogger('agent')
    extra_spec = {
        'core_state':
            hk.without_apply_rng(hk.transform(initial_state_fn)).apply(None),
        'logits':
            np.ones(shape=(num_actions,), dtype=np.float32)
    }
    reverb_queue = replay.make_reverb_online_queue(
        environment_spec=environment_spec,
        extra_spec=extra_spec,
        max_queue_size=max_queue_size,
        sequence_length=sequence_length,
        sequence_period=sequence_period,
        batch_size=batch_size,
    )
    self._server = reverb_queue.server
    self._can_sample = reverb_queue.can_sample

    # Make the learner.
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_gradient_norm),
        optax.adam(learning_rate),
    )
    self._learner = learning.IMPALALearner(
        obs_spec=environment_spec.observations,
        unroll_fn=unroll_fn,
        initial_state_fn=initial_state_fn,
        iterator=reverb_queue.data_iterator,
        rng=hk.PRNGSequence(seed),
        counter=counter,
        logger=logger,
        optimizer=optimizer,
        discount=discount,
        entropy_cost=entropy_cost,
        baseline_cost=baseline_cost,
        max_abs_reward=max_abs_reward,
    )

    # Make the actor.
    variable_client = variable_utils.VariableClient(self._learner, key='policy')
    transformed = hk.without_apply_rng(hk.transform(forward_fn))
    self._actor = acting.IMPALAActor(
        forward_fn=jax.jit(transformed.apply, backend='cpu'),
        initial_state_fn=initial_state_fn,
        rng=hk.PRNGSequence(seed),
        adder=reverb_queue.adder,
        variable_client=variable_client,
    )

  def observe_first(self, timestep: dm_env.TimeStep):
    self._actor.observe_first(timestep)

  def observe(
      self,
      action: types.Action,
      next_timestep: dm_env.TimeStep,
  ):
    self._actor.observe(action, next_timestep)

  def update(self, wait: bool = False):
    should_update_actor = False
    # Run a number of learner steps (usually gradient steps).
    while self._can_sample():
      self._learner.step()
      should_update_actor = True
    if should_update_actor:
      # Update actor weights after learner.
      self._actor.update(wait)

  def select_action(self, observation: np.ndarray) -> int:
    return self._actor.select_action(observation)
