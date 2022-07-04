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

"""DQN Builder."""
from typing import Iterator, List, Optional, Sequence

from acme import adders
from acme import core
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.agents.jax.dqn import actor as dqn_actor
from acme.agents.jax.dqn import config as dqn_config
from acme.agents.jax.dqn import learning_lib
from acme.datasets import reverb as datasets
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import jax
import optax
import reverb
from reverb import rate_limiters


class DQNBuilder(builders.ActorLearnerBuilder[networks_lib.FeedForwardNetwork,
                                              dqn_actor.EpsilonPolicy,
                                              reverb.ReplaySample]):
  """DQN Builder."""

  def __init__(self,
               config: dqn_config.DQNConfig,
               loss_fn: learning_lib.LossFn,
               actor_backend: Optional[str] = 'cpu'):
    """Creates DQN learner and the behavior policies.

    Args:
      config: DQN config.
      loss_fn: A loss function.
      actor_backend: Which backend to use when jitting the policy.
    """
    self._config = config
    self._loss_fn = loss_fn
    self._actor_backend = actor_backend

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: networks_lib.FeedForwardNetwork,
      dataset: Iterator[reverb.ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: Optional[specs.EnvironmentSpec],
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    del environment_spec

    return learning_lib.SGDLearner(
        network=networks,
        random_key=random_key,
        optimizer=optax.adam(
            self._config.learning_rate, eps=self._config.adam_eps),
        target_update_period=self._config.target_update_period,
        data_iterator=dataset,
        loss_fn=self._loss_fn,
        replay_client=replay_client,
        replay_table_name=self._config.replay_table_name,
        counter=counter,
        num_sgd_steps_per_step=self._config.num_sgd_steps_per_step,
        logger=logger_fn('learner'))

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy: dqn_actor.EpsilonPolicy,
      environment_spec: Optional[specs.EnvironmentSpec],
      variable_source: Optional[core.VariableSource] = None,
      adder: Optional[adders.Adder] = None,
  ) -> core.Actor:
    del environment_spec
    assert variable_source is not None
    # Inference happens on CPU, so it's better to move variables there too.
    variable_client = variable_utils.VariableClient(
        variable_source, '', device='cpu')
    epsilon = self._config.epsilon
    epsilons = epsilon if epsilon is Sequence else (epsilon,)
    actor_core = dqn_actor.alternating_epsilons_actor_core(
        policy, epsilons=epsilons)
    return actors.GenericActor(
        actor=actor_core,
        random_key=random_key,
        variable_client=variable_client,
        adder=adder,
        backend=self._actor_backend)

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy: dqn_actor.EpsilonPolicy,
  ) -> List[reverb.Table]:
    """Creates reverb tables for the algorithm."""
    del policy
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
            sampler=reverb.selectors.Prioritized(
                self._config.priority_exponent),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=limiter,
            signature=adders_reverb.NStepTransitionAdder.signature(
                environment_spec))
    ]

  def make_dataset_iterator(
      self, replay_client: reverb.Client) -> Iterator[reverb.ReplaySample]:
    """Creates a dataset iterator to use for learning."""
    dataset = datasets.make_reverb_dataset(
        table=self._config.replay_table_name,
        server_address=replay_client.server_address,
        batch_size=(self._config.batch_size *
                    self._config.num_sgd_steps_per_step),
        prefetch_size=self._config.prefetch_size)
    return utils.device_put(dataset.as_numpy_iterator(), jax.devices()[0])

  def make_adder(self, replay_client: reverb.Client) -> adders.Adder:
    """Creates an adder which handles observations."""
    return adders_reverb.NStepTransitionAdder(
        priority_fns={self._config.replay_table_name: None},
        client=replay_client,
        n_step=self._config.n_step,
        discount=self._config.discount)

  def make_policy(self,
                  networks: networks_lib.FeedForwardNetwork,
                  environment_spec: specs.EnvironmentSpec,
                  evaluation: bool = False) -> dqn_actor.EpsilonPolicy:
    """Creates the policy."""
    del environment_spec, evaluation
    return dqn_actor.behavior_policy(networks)
