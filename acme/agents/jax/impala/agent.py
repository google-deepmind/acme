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

from typing import Callable, Optional, Union

import acme
from acme import specs
from acme import types
from acme.agents import replay
from acme.agents.jax.impala import acting
from acme.agents.jax.impala import config as impala_config
from acme.agents.jax.impala import learning
from acme.agents.jax.impala import networks as impala_networks
from acme.agents.jax.impala import types as impala_types
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import dm_env
import haiku as hk
import jax
import numpy as np
import optax


class IMPALAFromConfig(acme.Actor):
  """IMPALA Agent."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      forward_fn: impala_types.PolicyValueFn,
      unroll_init_fn: impala_types.PolicyValueInitFn,
      unroll_fn: impala_types.PolicyValueFn,
      initial_state_init_fn: impala_types.RecurrentStateInitFn,
      initial_state_fn: impala_types.RecurrentStateFn,
      config: impala_config.IMPALAConfig,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
  ):
    networks = impala_networks.IMPALANetworks(
        forward_fn=forward_fn,
        unroll_init_fn=unroll_init_fn,
        unroll_fn=unroll_fn,
        initial_state_init_fn=initial_state_init_fn,
        initial_state_fn=initial_state_fn,
    )

    self._config = config

    # Data is handled by the reverb replay queue.
    num_actions = environment_spec.actions.num_values
    self._logger = logger or loggers.TerminalLogger('agent')

    key, key_initial_state = jax.random.split(
        jax.random.PRNGKey(self._config.seed))
    params = initial_state_init_fn(key_initial_state)
    extra_spec = {
        'core_state': initial_state_fn(params),
        'logits': np.ones(shape=(num_actions,), dtype=np.float32)
    }

    reverb_queue = replay.make_reverb_online_queue(
        environment_spec=environment_spec,
        extra_spec=extra_spec,
        max_queue_size=self._config.max_queue_size,
        sequence_length=self._config.sequence_length,
        sequence_period=self._config.sequence_period,
        batch_size=self._config.batch_size,
    )
    self._server = reverb_queue.server
    self._can_sample = reverb_queue.can_sample

    # Make the learner.
    optimizer = optax.chain(
        optax.clip_by_global_norm(self._config.max_gradient_norm),
        optax.adam(self._config.learning_rate),
    )
    key_learner, key_actor = jax.random.split(key)
    self._learner = learning.IMPALALearner(
        networks=networks,
        obs_spec=environment_spec.observations,
        iterator=reverb_queue.data_iterator,
        random_key=key_learner,
        counter=counter,
        logger=logger,
        optimizer=optimizer,
        discount=self._config.discount,
        entropy_cost=self._config.entropy_cost,
        baseline_cost=self._config.baseline_cost,
        max_abs_reward=self._config.max_abs_reward,
    )

    # Make the actor.
    variable_client = variable_utils.VariableClient(self._learner, key='policy')
    self._actor = acting.IMPALAActor(
        forward_fn=jax.jit(forward_fn, backend='cpu'),
        initial_state_init_fn=initial_state_init_fn,
        initial_state_fn=initial_state_fn,
        rng=hk.PRNGSequence(key_actor),
        adder=reverb_queue.adder,
        variable_client=variable_client,
    )

  def observe_first(self, timestep: dm_env.TimeStep):
    self._actor.observe_first(timestep)

  def observe(
      self,
      action: impala_types.Action,
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


class IMPALA(IMPALAFromConfig):
  """IMPALA agent.

  We are in the process of migrating towards a more modular agent configuration.
  This is maintained now for compatibility.
  """

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      forward_fn: networks_lib.PolicyValueRNN,
      unroll_fn: networks_lib.PolicyValueRNN,
      initial_state_fn: Callable[[], hk.LSTMState],
      sequence_length: int,
      sequence_period: int,
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      discount: float = 0.99,
      max_queue_size: Union[int, types.Batches] = types.Batches(10),
      batch_size: int = 16,
      learning_rate: float = 1e-3,
      entropy_cost: float = 0.01,
      baseline_cost: float = 0.5,
      seed: int = 0,
      max_abs_reward: float = np.inf,
      max_gradient_norm: float = np.inf,
  ):

    forward_fn_transformed = hk.without_apply_rng(hk.transform(
        forward_fn,
        apply_rng=True))
    unroll_fn_transformed = hk.without_apply_rng(hk.transform(
        unroll_fn,
        apply_rng=True))
    initial_state_fn_transformed = hk.without_apply_rng(hk.transform(
        initial_state_fn,
        apply_rng=True))

    config = impala_config.IMPALAConfig(
        sequence_length=sequence_length,
        sequence_period=sequence_period,
        discount=discount,
        max_queue_size=max_queue_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        entropy_cost=entropy_cost,
        baseline_cost=baseline_cost,
        seed=seed,
        max_abs_reward=max_abs_reward,
        max_gradient_norm=max_gradient_norm,
    )
    super().__init__(
        environment_spec=environment_spec,
        forward_fn=forward_fn_transformed.apply,
        unroll_init_fn=unroll_fn_transformed.init,
        unroll_fn=unroll_fn_transformed.apply,
        initial_state_init_fn=initial_state_fn_transformed.init,
        initial_state_fn=initial_state_fn_transformed.apply,
        config=config,
        counter=counter,
        logger=logger,
    )
