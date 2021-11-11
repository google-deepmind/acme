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

"""SAC Builder."""
from typing import Callable, Iterator, List, Optional

import acme
from acme import adders
from acme import core
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.agents.jax.sac import config as sac_config
from acme.agents.jax.sac import learning
from acme.agents.jax.sac import networks as sac_networks
from acme.datasets import reverb as datasets
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import optax
import reverb
from reverb import rate_limiters


class SACBuilder(builders.ActorLearnerBuilder):
  """SAC Builder."""

  def __init__(
      self,
      config: sac_config.SACConfig,
      logger_fn: Callable[[], Optional[loggers.Logger]] = lambda: None,
  ):
    """Creates a SAC learner, a behavior policy and an eval actor.

    Args:
      config: a config with SAC hps
      logger_fn: a logger factory for the learner
    """
    self._config = config
    self._logger_fn = logger_fn

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: sac_networks.SACNetworks,
      dataset: Iterator[reverb.ReplaySample],
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    # Create optimizers
    policy_optimizer = optax.adam(learning_rate=self._config.learning_rate)
    q_optimizer = optax.adam(learning_rate=self._config.learning_rate)

    return learning.SACLearner(
        networks=networks,
        tau=self._config.tau,
        discount=self._config.discount,
        entropy_coefficient=self._config.entropy_coefficient,
        target_entropy=self._config.target_entropy,
        rng=random_key,
        reward_scale=self._config.reward_scale,
        num_sgd_steps_per_step=self._config.num_sgd_steps_per_step,
        policy_optimizer=policy_optimizer,
        q_optimizer=q_optimizer,
        iterator=dataset,
        logger=self._logger_fn(),
        counter=counter)

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy_network,
      adder: Optional[adders.Adder] = None,
      variable_source: Optional[core.VariableSource] = None) -> acme.Actor:
    assert variable_source is not None
    actor_core = actor_core_lib.batched_feed_forward_to_actor_core(
        policy_network)
    variable_client = variable_utils.VariableClient(variable_source, 'policy',
                                                    device='cpu')
    return actors.GenericActor(
        actor_core, random_key, variable_client, adder, backend='cpu')

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
        batch_size=(
            self._config.batch_size * self._config.num_sgd_steps_per_step),
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
