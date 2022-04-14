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

"""DQN agent implementation."""

from typing import Sequence

from acme import specs
from acme.agents import agent
from acme.agents import replay
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax.dqn import config as dqn_config
from acme.agents.jax.dqn import learning_lib
from acme.agents.jax.dqn import losses
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
import jax
import jax.numpy as jnp
import optax
import rlax


class DQNFromConfig(agent.Agent):
  """DQN agent.

  This implements a single-process DQN agent. This is a simple Q-learning
  algorithm that inserts N-step transitions into a replay buffer, and
  periodically updates its policy by sampling these transitions using
  prioritization.
  """

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      network: networks_lib.FeedForwardNetwork,
      config: dqn_config.DQNConfig,
  ):
    """Initialize the agent."""
    # Data is communicated via reverb replay.
    reverb_replay = replay.make_reverb_prioritized_nstep_replay(
        environment_spec=environment_spec,
        n_step=config.n_step,
        batch_size=config.batch_size,
        max_replay_size=config.max_replay_size,
        min_replay_size=config.min_replay_size,
        priority_exponent=config.priority_exponent,
        discount=config.discount,
    )
    self._server = reverb_replay.server

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_gradient_norm),
        optax.adam(config.learning_rate),
    )
    key_learner, key_actor = jax.random.split(jax.random.PRNGKey(config.seed))
    # The learner updates the parameters (and initializes them).
    loss_fn = losses.PrioritizedDoubleQLearning(
        discount=config.discount,
        importance_sampling_exponent=config.importance_sampling_exponent,
    )
    learner = learning_lib.SGDLearner(
        network=network,
        loss_fn=loss_fn,
        data_iterator=reverb_replay.data_iterator,
        optimizer=optimizer,
        target_update_period=config.target_update_period,
        random_key=key_learner,
        replay_client=reverb_replay.client,
    )

    # The actor selects actions according to the policy.
    assert config.epsilon is not Sequence
    def policy(params: networks_lib.Params, key: jnp.ndarray,
               observation: jnp.ndarray) -> jnp.ndarray:
      action_values = network.apply(params, observation)
      return rlax.epsilon_greedy(config.epsilon).sample(key, action_values)
    actor_core = actor_core_lib.batched_feed_forward_to_actor_core(policy)
    variable_client = variable_utils.VariableClient(learner, '')
    actor = actors.GenericActor(
        actor_core, key_actor, variable_client, reverb_replay.adder)

    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=max(config.batch_size, config.min_replay_size),
        observations_per_step=config.batch_size / config.samples_per_insert,
    )


class DQN(DQNFromConfig):
  """DQN agent.

  We are in the process of migrating towards a more modular agent configuration.
  This is maintained now for compatibility.
  """

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      network: networks_lib.FeedForwardNetwork,
      batch_size: int = 256,
      prefetch_size: int = 4,
      target_update_period: int = 100,
      samples_per_insert: float = 0.5,
      min_replay_size: int = 1000,
      max_replay_size: int = 1000000,
      importance_sampling_exponent: float = 0.2,
      priority_exponent: float = 0.6,
      n_step: int = 5,
      epsilon: float = 0.05,
      learning_rate: float = 1e-3,
      discount: float = 0.99,
      seed: int = 1,
  ):
    config = dqn_config.DQNConfig(
        batch_size=batch_size,
        prefetch_size=prefetch_size,
        target_update_period=target_update_period,
        samples_per_insert=samples_per_insert,
        min_replay_size=min_replay_size,
        max_replay_size=max_replay_size,
        importance_sampling_exponent=importance_sampling_exponent,
        priority_exponent=priority_exponent,
        n_step=n_step,
        epsilon=epsilon,
        learning_rate=learning_rate,
        discount=discount,
        seed=seed,
    )
    super().__init__(
        environment_spec=environment_spec,
        network=network,
        config=config,
    )
