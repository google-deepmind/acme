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

"""DQN agent implementation."""

from acme import datasets
from acme import specs
from acme.adders import reverb as adders
from acme.agents import actors_jax
from acme.agents import agent
from acme.agents.jax.dqn import learning
from acme.networks import jax as networks
from acme.utils import jax_variable_utils

import haiku as hk
from jax.experimental import optix
import jax.numpy as jnp
import reverb
import rlax


class DQN(agent.Agent):
  """DQN agent.

  This implements a single-process DQN agent. This is a simple Q-learning
  algorithm that inserts N-step transitions into a replay buffer, and
  periodically updates its policy by sampling these transitions using
  prioritization.
  """

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      network: networks.QNetwork,
      batch_size: int = 256,
      prefetch_size: int = 4,
      target_update_period: int = 100,
      samples_per_insert: float = 32.0,
      min_replay_size: int = 1000,
      max_replay_size: int = 1000000,
      importance_sampling_exponent: float = 0.2,
      priority_exponent: float = 0.6,
      n_step: int = 5,
      epsilon: float = 0.,
      learning_rate: float = 1e-3,
      discount: float = 0.99,
  ):
    """Initialize the agent."""

    # Create a replay server to add data to. This uses no limiter behavior in
    # order to allow the Agent interface to handle it.
    replay_table = reverb.Table(
        name=adders.DEFAULT_PRIORITY_TABLE,
        sampler=reverb.selectors.Prioritized(priority_exponent),
        remover=reverb.selectors.Fifo(),
        max_size=max_replay_size,
        rate_limiter=reverb.rate_limiters.MinSize(1))
    self._server = reverb.Server([replay_table], port=None)

    # The adder is used to insert observations into replay.
    address = f'localhost:{self._server.port}'
    adder = adders.NStepTransitionAdder(
        client=reverb.Client(address),
        n_step=n_step,
        discount=discount)

    # The dataset provides an interface to sample from replay.
    dataset = datasets.make_reverb_dataset(
        client=reverb.TFClient(address),
        environment_spec=environment_spec,
        batch_size=batch_size,
        prefetch_size=prefetch_size,
        transition_adder=True)

    def policy(params: hk.Params, key: jnp.ndarray,
               observation: jnp.ndarray) -> jnp.ndarray:
      action_values = hk.transform(network).apply(params, observation)
      return rlax.epsilon_greedy(epsilon).sample(key, action_values)

    # The learner updates the parameters (and initializes them).
    learner = learning.DQNLearner(
        network=network,
        obs_spec=environment_spec.observations,
        rng=hk.PRNGSequence(1),
        optimizer=optix.adam(learning_rate),
        discount=discount,
        importance_sampling_exponent=importance_sampling_exponent,
        target_update_period=target_update_period,
        iterator=dataset.as_numpy_iterator(),
        replay_client=reverb.Client(address),
    )

    variable_client = jax_variable_utils.VariableClient(learner, 'foo')

    actor = actors_jax.FeedForwardActor(
        policy=policy,
        rng=hk.PRNGSequence(1),
        variable_client=variable_client,
        adder=adder)

    super().__init__(
        actor=actor,
        learner=learner,
        min_observations=max(batch_size, min_replay_size),
        observations_per_step=float(batch_size) / samples_per_insert)
