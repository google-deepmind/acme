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

"""IMPALA Builder."""

from typing import Any, Callable, Iterator, List, Optional

import acme
from acme import adders
from acme import core
from acme import specs
from acme.adders import reverb as reverb_adders
from acme.agents.jax import builders
from acme.agents.jax.impala import acting
from acme.agents.jax.impala import config as impala_config
from acme.agents.jax.impala import learning
from acme.agents.jax.impala import networks as impala_networks
from acme.datasets import reverb as datasets
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import reverb


class IMPALABuilder(builders.ActorLearnerBuilder[impala_networks.IMPALANetworks,
                                                 impala_networks.IMPALANetworks,
                                                 reverb.ReplaySample]):
  """IMPALA Builder."""

  def __init__(
      self,
      config: impala_config.IMPALAConfig,
      core_state_spec: hk.LSTMState,
      table_extension: Optional[Callable[[], Any]] = None,
  ):
    """Creates an IMPALA learner."""
    self._config = config
    self._core_state_spec = core_state_spec
    self._sequence_length = self._config.sequence_length
    self._table_extension = table_extension

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy: impala_networks.IMPALANetworks,
  ) -> List[reverb.Table]:
    """The queue; use XData or INFO log."""
    del policy
    num_actions = environment_spec.actions.num_values
    extra_spec = {
        'core_state': self._core_state_spec,
        'logits': jnp.ones(shape=(num_actions,), dtype=jnp.float32)
    }
    signature = reverb_adders.SequenceAdder.signature(
        environment_spec,
        extra_spec,
        sequence_length=self._config.sequence_length)

    # Maybe create rate limiter.
    # Setting the samples_per_insert ratio less than the default of 1.0, allows
    # the agent to drop data for the benefit of using data from most up-to-date
    # policies to compute its learner updates.
    samples_per_insert = self._config.samples_per_insert
    if samples_per_insert:
      if samples_per_insert > 1.0 or samples_per_insert <= 0.0:
        raise ValueError(
            'Impala requires a samples_per_insert ratio in the range (0, 1],'
            f' but received {samples_per_insert}.')
      limiter = reverb.rate_limiters.SampleToInsertRatio(
          samples_per_insert=samples_per_insert,
          min_size_to_sample=1,
          error_buffer=self._config.batch_size)
    else:
      limiter = reverb.rate_limiters.MinSize(1)

    table_extensions = []
    if self._table_extension is not None:
      table_extensions = [self._table_extension()]
    queue = reverb.Table(
        name=self._config.replay_table_name,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=self._config.max_queue_size,
        max_times_sampled=1,
        rate_limiter=limiter,
        extensions=table_extensions,
        signature=signature)
    return [queue]

  def make_dataset_iterator(
      self, replay_client: reverb.Client) -> Iterator[reverb.ReplaySample]:
    """Creates a dataset."""
    batch_size_per_learner = self._config.batch_size // jax.process_count()
    batch_size_per_device, ragged = divmod(self._config.batch_size,
                                           jax.device_count())
    if ragged:
      raise ValueError(
          'Learner batch size must be divisible by total number of devices!')

    dataset = datasets.make_reverb_dataset(
        table=self._config.replay_table_name,
        server_address=replay_client.server_address,
        batch_size=batch_size_per_device,
        num_parallel_calls=None,
        max_in_flight_samples_per_worker=2 * batch_size_per_learner)

    return utils.multi_device_put(dataset.as_numpy_iterator(),
                                  jax.local_devices())

  def make_adder(self, replay_client: reverb.Client) -> adders.Adder:
    """Creates an adder which handles observations."""
    # Note that the last transition in the sequence is used for bootstrapping
    # only and is ignored otherwise. So we need to make sure that sequences
    # overlap on one transition, thus "-1" in the period length computation.
    return reverb_adders.SequenceAdder(
        client=replay_client,
        priority_fns={self._config.replay_table_name: None},
        period=self._config.sequence_period or (self._sequence_length - 1),
        sequence_length=self._sequence_length,
    )

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: impala_networks.IMPALANetworks,
      dataset: Iterator[reverb.ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    del environment_spec, replay_client

    optimizer = optax.chain(
        optax.clip_by_global_norm(self._config.max_gradient_norm),
        optax.adam(
            self._config.learning_rate,
            b1=self._config.adam_momentum_decay,
            b2=self._config.adam_variance_decay),
    )

    return learning.IMPALALearner(
        networks=networks,
        iterator=dataset,
        optimizer=optimizer,
        random_key=random_key,
        discount=self._config.discount,
        entropy_cost=self._config.entropy_cost,
        baseline_cost=self._config.baseline_cost,
        max_abs_reward=self._config.max_abs_reward,
        counter=counter,
        logger=logger_fn('learner'),
    )

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy: impala_networks.IMPALANetworks,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
      adder: Optional[adders.Adder] = None,
  ) -> acme.Actor:
    del environment_spec
    variable_client = variable_utils.VariableClient(
        client=variable_source, key='network', update_period=1000, device='cpu')
    return acting.IMPALAActor(
        forward_fn=policy.forward_fn,
        initial_state_fn=policy.initial_state_fn,
        variable_client=variable_client,
        adder=adder,
        rng=hk.PRNGSequence(random_key),
    )

  def make_policy(
      self,
      networks: impala_networks.IMPALANetworks[Any],
      environment_spec: specs.EnvironmentSpec,
      evaluation: bool = False) -> impala_networks.IMPALANetworks[Any]:
    del environment_spec, evaluation
    return networks
