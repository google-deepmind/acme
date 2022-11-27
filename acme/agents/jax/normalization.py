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

"""Utility classes for input normalization."""

import dataclasses
import functools
from typing import Any, Callable, Generic, Iterator, List, Optional, Tuple

import acme
from acme import adders
from acme import core
from acme import specs
from acme import types
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import running_statistics
from acme.jax import variable_utils
from acme.jax.types import Networks, Policy  # pylint: disable=g-multiple-import
from acme.utils import counting
from acme.utils import loggers
import dm_env
import jax
import reverb
from typing_extensions import Protocol

_NORMALIZATION_VARIABLES = 'normalization_variables'


# Wrapping the network instead might look more straightforward, but then
# different implementations would be needed for feed-forward and
# recurrent networks.
class NormalizationActorWrapper(core.Actor):
  """An actor wrapper that normalizes observations before applying policy."""

  def __init__(self,
               wrapped_actor: core.Actor,
               variable_source: core.VariableSource,
               max_abs_observation: Optional[float],
               update_period: int = 1,
               backend: Optional[str] = None):
    self._wrapped_actor = wrapped_actor
    self._variable_client = variable_utils.VariableClient(
        variable_source,
        key=_NORMALIZATION_VARIABLES,
        update_period=update_period,
        device=backend)
    self._apply_normalization = jax.jit(
        functools.partial(
            running_statistics.normalize, max_abs_value=max_abs_observation),
        backend=backend)

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    self._variable_client.update()
    observation_stats = self._variable_client.params
    observation = self._apply_normalization(observation, observation_stats)
    return self._wrapped_actor.select_action(observation)

  def observe_first(self, timestep: dm_env.TimeStep):
    return self._wrapped_actor.observe_first(timestep)

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    return self._wrapped_actor.observe(action, next_timestep)

  def update(self, wait: bool = False):
    return self._wrapped_actor.update(wait)


@dataclasses.dataclass
class NormalizationLearnerWrapperState:
  wrapped_learner_state: Any
  observation_running_statistics: running_statistics.RunningStatisticsState


class NormalizationLearnerWrapper(core.Learner, core.Saveable):
  """A learner wrapper that normalizes observations using running statistics."""

  def __init__(self, learner_factory: Callable[[Iterator[reverb.ReplaySample]],
                                               acme.Learner],
               iterator: Iterator[reverb.ReplaySample],
               environment_spec: specs.EnvironmentSpec,
               max_abs_observation: Optional[float]):

    def normalize_sample(
        observation_statistics: running_statistics.RunningStatisticsState,
        sample: reverb.ReplaySample
    ) -> Tuple[running_statistics.RunningStatisticsState, reverb.ReplaySample]:
      observation = sample.data.observation
      observation_statistics = running_statistics.update(
          observation_statistics, observation)
      observation = running_statistics.normalize(
          observation,
          observation_statistics,
          max_abs_value=max_abs_observation)
      sample = reverb.ReplaySample(
          sample.info, sample.data._replace(observation=observation))
      if hasattr(sample.data, 'next_observation'):
        next_observation = running_statistics.normalize(
            sample.data.next_observation,
            observation_statistics,
            max_abs_value=max_abs_observation)
        sample = reverb.ReplaySample(
            sample.info,
            sample.data._replace(next_observation=next_observation))

      return observation_statistics, sample

    self._observation_running_statistics = running_statistics.init_state(
        environment_spec.observations)
    self._normalize_sample = jax.jit(normalize_sample)

    normalizing_iterator = (
        self._normalize_sample_and_update(sample) for sample in iterator)
    self._wrapped_learner = learner_factory(normalizing_iterator)

  def _normalize_sample_and_update(
      self, sample: reverb.ReplaySample) -> reverb.ReplaySample:
    self._observation_running_statistics, sample = self._normalize_sample(
        self._observation_running_statistics, sample)
    return sample

  def step(self):
    self._wrapped_learner.step()

  def get_variables(self, names: List[str]) -> List[types.NestedArray]:
    stats = self._observation_running_statistics
    # Make sure to only pass mean and std to minimize trafic.
    mean_std = running_statistics.NestedMeanStd(mean=stats.mean, std=stats.std)
    normalization_variables = {_NORMALIZATION_VARIABLES: mean_std}

    learner_names = [
        name for name in names if name not in normalization_variables
    ]
    learner_variables = dict(
        zip(learner_names, self._wrapped_learner.get_variables(
            learner_names))) if learner_names else {}

    return [
        normalization_variables.get(name, learner_variables.get(name, None))
        for name in names
    ]

  def save(self) -> NormalizationLearnerWrapperState:
    return NormalizationLearnerWrapperState(
        wrapped_learner_state=self._wrapped_learner.save(),
        observation_running_statistics=self._observation_running_statistics)

  def restore(self, state: NormalizationLearnerWrapperState):
    self._wrapped_learner.restore(state.wrapped_learner_state)
    self._observation_running_statistics = state.observation_running_statistics


@dataclasses.dataclass
class NormalizationBuilder(Generic[Networks, Policy],
                           builders.ActorLearnerBuilder[Networks, Policy,
                                                        reverb.ReplaySample]):
  """Builder wrapper that normalizes observations using running mean/std."""
  builder: builders.ActorLearnerBuilder[Networks, Policy, reverb.ReplaySample]
  max_abs_observation: Optional[float] = 10.0
  statistics_update_period: int = 100

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy: Policy,
  ) -> List[reverb.Table]:
    return self.builder.make_replay_tables(environment_spec, policy)

  def make_dataset_iterator(
      self, replay_client: reverb.Client) -> Iterator[reverb.ReplaySample]:
    return self.builder.make_dataset_iterator(replay_client)

  def make_adder(self, replay_client: reverb.Client,
                 environment_spec: Optional[specs.EnvironmentSpec],
                 policy: Optional[Policy]) -> Optional[adders.Adder]:
    return self.builder.make_adder(replay_client, environment_spec, policy)

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: Networks,
      dataset: Iterator[reverb.ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:

    learner_factory = functools.partial(
        self.builder.make_learner,
        random_key,
        networks,
        logger_fn=logger_fn,
        environment_spec=environment_spec,
        replay_client=replay_client,
        counter=counter)

    return NormalizationLearnerWrapper(
        learner_factory=learner_factory,
        iterator=dataset,
        environment_spec=environment_spec,
        max_abs_observation=self.max_abs_observation)

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy: Policy,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
      adder: Optional[adders.Adder] = None,
  ) -> core.Actor:
    actor = self.builder.make_actor(random_key, policy, environment_spec,
                                    variable_source, adder)
    return NormalizationActorWrapper(
        actor,
        variable_source,
        max_abs_observation=self.max_abs_observation,
        update_period=self.statistics_update_period,
        backend='cpu')

  def make_policy(self,
                  networks: Networks,
                  environment_spec: specs.EnvironmentSpec,
                  evaluation: bool = False) -> Policy:
    return self.builder.make_policy(
        networks=networks,
        environment_spec=environment_spec,
        evaluation=evaluation)


@dataclasses.dataclass(frozen=True)
class NormalizationConfig:
  """Configuration for normalization based on running statistics.

  Attributes:
    max_abs: Maximum value for clipping.
    statistics_update_period: How often to update running statistics used for
      normalization.
  """
  max_abs: int = 10
  statistics_update_period: int = 100


class InputNormalizerConfig(Protocol):
  """Protocol for the config of the agent that uses the normalization decorator.

  If the agent builder is decorated with the `input_normalization_builder`
  the agent config class must implement this protocol.
  """

  @property
  def input_normalization(self) -> Optional[NormalizationConfig]:
    ...


def input_normalization_builder(
    actor_learner_builder_class: Callable[[InputNormalizerConfig],
                                          builders.ActorLearnerBuilder]):
  """Builder class decorator that adds support for input normalization."""

  # TODO(b/247075349): find a way to use ActorLearnerBuilderWrapper here.
  class InputNormalizationBuilder(
      Generic[builders.Networks, builders.Policy, builders.Sample],
      builders.ActorLearnerBuilder[builders.Networks, builders.Policy,
                                   builders.Sample]):
    """Builder wrapper that adds input normalization based on the config."""

    def __init__(self, config: InputNormalizerConfig):
      builder = actor_learner_builder_class(config)
      if config.input_normalization:
        builder = NormalizationBuilder(
            builder,
            max_abs_observation=config.input_normalization.max_abs,
            statistics_update_period=config.input_normalization
            .statistics_update_period)
      self.wrapped = builder

    def make_replay_tables(
        self,
        environment_spec: specs.EnvironmentSpec,
        policy: builders.Policy,
    ) -> List[reverb.Table]:
      return self.wrapped.make_replay_tables(environment_spec, policy)

    def make_dataset_iterator(
        self,
        replay_client: reverb.Client,
    ) -> Iterator[builders.Sample]:
      return self.wrapped.make_dataset_iterator(replay_client)

    def make_adder(
        self,
        replay_client: reverb.Client,
        environment_spec: Optional[specs.EnvironmentSpec],
        policy: Optional[builders.Policy],
    ) -> Optional[adders.Adder]:
      return self.wrapped.make_adder(replay_client, environment_spec, policy)

    def make_actor(
        self,
        random_key: networks_lib.PRNGKey,
        policy: builders.Policy,
        environment_spec: specs.EnvironmentSpec,
        variable_source: Optional[core.VariableSource] = None,
        adder: Optional[adders.Adder] = None,
    ) -> core.Actor:
      return self.wrapped.make_actor(random_key, policy, environment_spec,
                                     variable_source, adder)

    def make_learner(
        self,
        random_key: networks_lib.PRNGKey,
        networks: Networks,
        dataset: Iterator[builders.Sample],
        logger_fn: loggers.LoggerFactory,
        environment_spec: specs.EnvironmentSpec,
        replay_client: Optional[reverb.Client] = None,
        counter: Optional[counting.Counter] = None,
    ) -> core.Learner:
      return self.wrapped.make_learner(random_key, networks, dataset, logger_fn,
                                       environment_spec, replay_client, counter)

    def make_policy(self,
                    networks: builders.Networks,
                    environment_spec: specs.EnvironmentSpec,
                    evaluation: bool = False) -> builders.Policy:
      return self.wrapped.make_policy(networks, environment_spec, evaluation)

  return InputNormalizationBuilder
