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

"""D4PG Builder."""
import dataclasses
from typing import Callable, Iterator, List, Optional

import acme
from acme import adders
from acme import core
from acme import specs
from acme import types
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.agents.jax.d4pg import learning
from acme.datasets import reverb as datasets
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import optax
import reverb
from reverb import rate_limiters
import rlax


@dataclasses.dataclass
class D4PGConfig:
  """Configuration options for D4PG."""
  sigma: float = 0.3
  target_update_period: int = 100
  samples_per_insert: float = 32.0

  # Loss options
  n_step: int = 5
  discount: float = 0.99
  batch_size: int = 256
  learning_rate: float = 1e-4
  clipping: bool = True

  # Replay options
  min_replay_size: int = 1000
  max_replay_size: int = 1000000
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
  prefetch_size: int = 4
  # Rate to be used for the SampleToInsertRatio rate limitter tolerance.
  # See a formula in make_replay_tables for more details.
  samples_per_insert_tolerance_rate: float = 0.1

  # How many gradient updates to perform per step.
  num_sgd_steps_per_step: int = 1


@dataclasses.dataclass
class D4PGNetworks:
  """Network and pure functions for the D4PG agent.."""
  policy_network: networks_lib.FeedForwardNetwork
  critic_network: networks_lib.FeedForwardNetwork


def get_default_behavior_policy(networks: D4PGNetworks, config: D4PGConfig):
  # Selects action according to the policy
  def behavior_policy(params: networks_lib.Params, key: networks_lib.PRNGKey,
                      observation: types.NestedArray):
    action = networks.policy_network.apply(params, observation)
    noisy_action = rlax.add_gaussian_noise(key, action, config.sigma)
    return noisy_action
  return behavior_policy


def get_default_eval_policy(networks: D4PGNetworks):
  # Selects action according to the policy
  def behavior_policy(params: networks_lib.Params, key: networks_lib.PRNGKey,
                      observation: types.NestedArray):
    del key  # Evaluation policy is deterministic.
    action = networks.policy_network.apply(params, observation)
    return action
  return behavior_policy


class D4PGBuilder(builders.ActorLearnerBuilder):
  """D4PG Builder."""

  def __init__(
      self,
      config: D4PGConfig,
      logger_fn: Callable[[], loggers.Logger] = lambda: None,
  ):
    """Creates a D4PG learner, a behavior policy and an eval actor.

    Args:
      config: a config with D4PG hps
      logger_fn: a logger factory for the learner
    """
    self._config = config
    self._logger_fn = logger_fn

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: D4PGNetworks,
      dataset: Iterator[reverb.ReplaySample],
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    policy_optimizer = optax.adam(self._config.learning_rate)
    critic_optimizer = optax.adam(self._config.learning_rate)

    if self._config.clipping:
      policy_optimizer = optax.chain(
          optax.clip_by_global_norm(40.), policy_optimizer)
      critic_optimizer = optax.chain(
          optax.clip_by_global_norm(40.), critic_optimizer)

    # The learner updates the parameters (and initializes them).
    return learning.D4PGLearner(
        policy_network=networks.policy_network,
        critic_network=networks.critic_network,
        random_key=random_key,
        policy_optimizer=policy_optimizer,
        critic_optimizer=critic_optimizer,
        clipping=self._config.clipping,
        discount=self._config.discount,
        target_update_period=self._config.target_update_period,
        iterator=dataset,
        counter=counter,
        logger=self._logger_fn(),
        num_sgd_steps_per_step=self._config.num_sgd_steps_per_step)

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
  ) -> List[reverb.Table]:
    """Create tables to insert data into."""
    samples_per_insert_tolerance = (
        self._config.samples_per_insert_tolerance_rate *
        self._config.samples_per_insert)
    error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
    limiter = rate_limiters.SampleToInsertRatio(
        min_size_to_sample=self._config.min_replay_size,
        samples_per_insert=self._config.samples_per_insert,
        error_buffer=error_buffer)
    return [
        reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=limiter,
            signature=adders_reverb.NStepTransitionAdder.signature(
                environment_spec))
    ]

  def make_dataset_iterator(
      self, replay_client: reverb.Client) -> Iterator[reverb.ReplaySample]:
    """Create a dataset iterator to use for learning/updating the agent."""
    dataset = datasets.make_reverb_dataset(
        table=self._config.replay_table_name,
        server_address=replay_client.server_address,
        batch_size=self._config.batch_size *
        self._config.num_sgd_steps_per_step,
        prefetch_size=self._config.prefetch_size)
    return dataset.as_numpy_iterator()

  def make_adder(self,
                 replay_client: reverb.Client) -> Optional[adders.Adder]:
    """Create an adder which records data generated by the actor/environment."""
    return adders_reverb.NStepTransitionAdder(
        priority_fns={self._config.replay_table_name: None},
        client=replay_client,
        n_step=self._config.n_step,
        discount=self._config.discount)

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy_network,
      adder: Optional[adders.Adder] = None,
      variable_source: Optional[core.VariableSource] = None) -> acme.Actor:
    assert variable_source is not None
    actor_core = actor_core_lib.batched_feed_forward_to_actor_core(
        policy_network)
    # Inference happens on CPU, so it's better to move variables there too.
    variable_client = variable_utils.VariableClient(variable_source, 'policy',
                                                    device='cpu')
    return actors.GenericActor(
        actor_core, random_key, variable_client, adder, backend='cpu')
