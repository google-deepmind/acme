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

"""PPO Builder."""
from typing import Callable, Iterator, List, Optional

from acme import adders
from acme import core
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.agents.jax.ppo import config as ppo_config
from acme.agents.jax.ppo import learning
from acme.agents.jax.ppo import networks as ppo_networks
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import optax
import reverb


class PPOBuilder(builders.ActorLearnerBuilder):
  """PPO Builder."""

  def __init__(
      self,
      config: ppo_config.PPOConfig,
      logger_fn: Callable[[], loggers.Logger] = lambda: None,
  ):
    """Creates PPO builder."""
    self._config = config
    self._logger_fn = logger_fn

    # An extra step is used for bootstrapping when computing advantages.
    self._sequence_length = config.unroll_length + 1

  def make_replay_tables(
      self, environment_spec: specs.EnvironmentSpec) -> List[reverb.Table]:
    """Creates reverb tables for the algorithm."""
    extra_spec = {
        'log_prob': np.ones(shape=(), dtype=np.float32),
    }
    signature = adders_reverb.SequenceAdder.signature(
        environment_spec, extra_spec, sequence_length=self._sequence_length)
    # We set a huge size of queue, even though it is not really necessary since
    # a custom update code in LocalLayout will make sure that we process data
    # as soon as it is available. So the size here does not really matter, as
    # long as it is bigger than or equal to the batch size.
    # CAVEAT: we cannot make it equal to batch_size or batch_size + 1, the value
    # has to be larger for few reasons:
    #  - during the last step in an episode we might write 2 sequences to reverb
    #    at once (that's how SequenceAdder works)
    #  - reverb does addition/sampling in multiple threads, so data might be
    #    added not when you expect it to be. so we need some extra buffer size
    #    in order not to get deadlocks.
    # TODO(raveman): This (large queue size) will not work for distributed setup
    # if learner is slower than actors (the queue will fill up and the learner
    # will be offpolicy). Find a better solution.
    is_single_machine = True
    extra_capacity_to_avoid_single_machine_deadlocks = 1000
    max_size = self._config.batch_size + (
        extra_capacity_to_avoid_single_machine_deadlocks
        if is_single_machine else 0)
    return [
        reverb.Table.queue(
            name=self._config.replay_table_name,
            max_size=max_size,
            signature=signature)
    ]

  def make_dataset_iterator(
      self, replay_client: reverb.Client) -> Iterator[reverb.ReplaySample]:
    """Creates a dataset."""
    # We don't use datasets.make_reverb_dataset() here to avoid interleaving
    # and prefetching, that doesn't work well with can_sample() check on update.
    # NOTE: Value for max_in_flight_samples_per_worker comes from a
    # recommendation here: https://git.io/JYzXB
    # TODO(bshahr): use TrajectoryDataset instead when the adders are updated.
    dataset = reverb.TrajectoryDataset.from_table_signature(
        server_address=replay_client.server_address,
        table=self._config.replay_table_name,
        max_in_flight_samples_per_worker=2 * self._config.batch_size)
    # Add batch dimension.
    dataset = dataset.batch(self._config.batch_size, drop_remainder=True)
    return dataset.as_numpy_iterator()

  def make_adder(self, replay_client: reverb.Client) -> adders.Adder:
    """Creates an adder which handles observations."""
    # Note that the last transition in the sequence is used for bootstrapping
    # only and is ignored otherwise. So we need to make sure that sequences
    # overlap on one transition, thus "-1" in the period length computation.
    return adders_reverb.SequenceAdder(
        client=replay_client,
        priority_fns={self._config.replay_table_name: None},
        period=self._sequence_length - 1,
        sequence_length=self._sequence_length,
        pad_end_of_episode=False,
        break_end_of_episode=False,
    )

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: ppo_networks.PPONetworks,
      dataset: Iterator[reverb.ReplaySample],
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    if callable(self._config.learning_rate):
      optimizer = optax.chain(
          optax.clip_by_global_norm(self._config.max_gradient_norm),
          optax.scale_by_adam(eps=self._config.adam_epsilon),
          optax.scale_by_schedule(self._config.learning_rate), optax.scale(-1))
    else:
      optimizer = optax.chain(
          optax.clip_by_global_norm(self._config.max_gradient_norm),
          optax.scale_by_adam(eps=self._config.adam_epsilon),
          optax.scale(-self._config.learning_rate))

    return learning.PPOLearner(
        ppo_networks=networks,
        iterator=dataset,
        discount=self._config.discount,
        entropy_cost=self._config.entropy_cost,
        value_cost=self._config.value_cost,
        max_abs_reward=self._config.max_abs_reward,
        ppo_clipping_epsilon=self._config.ppo_clipping_epsilon,
        clip_value=self._config.clip_value,
        gae_lambda=self._config.gae_lambda,
        counter=counter,
        random_key=random_key,
        optimizer=optimizer,
        num_epochs=self._config.num_epochs,
        num_minibatches=self._config.num_minibatches,
        logger=self._logger_fn(),
    )

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy_network,
      adder: Optional[adders.Adder] = None,
      variable_source: Optional[core.VariableSource] = None,
  ) -> core.Actor:
    assert variable_source is not None
    actor = actor_core_lib.batched_feed_forward_with_extras_to_actor_core(
        policy_network)
    variable_client = variable_utils.VariableClient(
        variable_source,
        'network',
        device='cpu',
        update_period=self._config.variable_update_period)
    return actors.GenericActor(
        actor, random_key, variable_client, adder, backend='cpu')
