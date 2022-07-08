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

"""R2D2 Builder."""
from typing import Generic, Iterator, List, Optional

import acme
from acme import adders
from acme import core
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.agents.jax.r2d2 import actor as r2d2_actor
from acme.agents.jax.r2d2 import config as r2d2_config
from acme.agents.jax.r2d2 import learning as r2d2_learning
from acme.agents.jax.r2d2 import networks as r2d2_networks
from acme.datasets import reverb as datasets
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import jax
import optax
import reverb


class R2D2Builder(Generic[actor_core_lib.RecurrentState],
                  builders.ActorLearnerBuilder[r2d2_networks.R2D2Networks,
                                               r2d2_actor.R2D2Policy,
                                               reverb.ReplaySample]):
  """R2D2 Builder.

  This is constructs all of the components for Recurrent Experience Replay in
  Distributed Reinforcement Learning (Kapturowski et al.)
  https://openreview.net/pdf?id=r1lyTjAqYX.
  """

  def __init__(self, config: r2d2_config.R2D2Config):
    """Creates a R2D2 learner, a behavior policy and an eval actor."""
    self._config = config
    self._sequence_length = (
        self._config.burn_in_length + self._config.trace_length + 1)

  @property
  def _batch_size_per_device(self) -> int:
    """Splits batch size across all learner devices evenly."""
    # TODO(bshahr): Using jax.device_count will not be valid when colocating
    # learning and inference.
    return self._config.batch_size // jax.device_count()

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: r2d2_networks.R2D2Networks,
      dataset: Iterator[r2d2_learning.R2D2ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    del environment_spec

    # The learner updates the parameters (and initializes them).
    return r2d2_learning.R2D2Learner(
        unroll=networks.unroll,
        initial_state=networks.initial_state,
        batch_size=self._batch_size_per_device,
        random_key=random_key,
        burn_in_length=self._config.burn_in_length,
        discount=self._config.discount,
        importance_sampling_exponent=(
            self._config.importance_sampling_exponent),
        max_priority_weight=self._config.max_priority_weight,
        target_update_period=self._config.target_update_period,
        iterator=dataset,
        optimizer=optax.adam(self._config.learning_rate),
        bootstrap_n=self._config.bootstrap_n,
        tx_pair=self._config.tx_pair,
        clip_rewards=self._config.clip_rewards,
        replay_client=replay_client,
        counter=counter,
        logger=logger_fn('learner'))

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy: r2d2_actor.R2D2Policy,
  ) -> List[reverb.Table]:
    """Create tables to insert data into."""
    dummy_actor_state = policy.init(jax.random.PRNGKey(0))
    extras_spec = policy.get_extras(dummy_actor_state)

    if self._config.samples_per_insert:
      samples_per_insert_tolerance = (
          self._config.samples_per_insert_tolerance_rate *
          self._config.samples_per_insert)
      error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
      limiter = reverb.rate_limiters.SampleToInsertRatio(
          min_size_to_sample=self._config.min_replay_size,
          samples_per_insert=self._config.samples_per_insert,
          error_buffer=error_buffer)
    else:
      limiter = reverb.rate_limiters.MinSize(self._config.min_replay_size)
    return [
        reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Prioritized(
                self._config.priority_exponent),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=limiter,
            signature=adders_reverb.SequenceAdder.signature(
                environment_spec, extras_spec))
    ]

  def make_dataset_iterator(
      self, replay_client: reverb.Client) -> Iterator[reverb.ReplaySample]:
    """Create a dataset iterator to use for learning/updating the agent."""
    dataset = datasets.make_reverb_dataset(
        table=self._config.replay_table_name,
        server_address=replay_client.server_address,
        batch_size=self._batch_size_per_device,
        prefetch_size=self._config.prefetch_size,
        num_parallel_calls=self._config.num_parallel_calls)

    # We split samples in two outputs, the keys which need to be kept on-host
    # since int64 arrays are not supported in TPUs, and the entire sample
    # separately so it can be sent to the sgd_step method.
    def split_sample(sample: reverb.ReplaySample) -> utils.PrefetchingSplit:
      return utils.PrefetchingSplit(host=sample.info.key, device=sample)

    return utils.multi_device_put(
        dataset.as_numpy_iterator(),
        devices=jax.local_devices(),
        split_fn=split_sample)

  def make_adder(self,
                 replay_client: reverb.Client) -> Optional[adders.Adder]:
    """Create an adder which records data generated by the actor/environment."""
    return adders_reverb.SequenceAdder(
        client=replay_client,
        period=self._config.sequence_period,
        sequence_length=self._sequence_length,
        delta_encoded=True)

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy: r2d2_actor.R2D2Policy,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
      adder: Optional[adders.Adder] = None,
  ) -> acme.Actor:
    del environment_spec
    # Create variable client.
    variable_client = variable_utils.VariableClient(
        variable_source,
        key='actor_variables',
        update_period=self._config.variable_update_period)

    return actors.GenericActor(
        policy, random_key, variable_client, adder, backend='cpu')

  def make_policy(self,
                  networks: r2d2_networks.R2D2Networks,
                  environment_spec: specs.EnvironmentSpec,
                  evaluation: bool = False) -> r2d2_actor.R2D2Policy:
    if evaluation:
      return r2d2_actor.get_actor_core(
          networks,
          num_epsilons=None,
          evaluation_epsilon=self._config.evaluation_epsilon)
    else:
      return r2d2_actor.get_actor_core(networks, self._config.num_epsilons)
