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

"""MBOP Builder."""
import functools
from typing import Iterator, Optional

from acme import core
from acme import specs
from acme import types
from acme.agents.jax import builders
from acme.agents.jax.mbop import acting
from acme.agents.jax.mbop import config as mbop_config
from acme.agents.jax.mbop import learning
from acme.agents.jax.mbop import losses as mbop_losses
from acme.agents.jax.mbop import networks as mbop_networks
from acme.jax import networks as networks_lib
from acme.jax import running_statistics
from acme.utils import counting
from acme.utils import loggers
import optax


class MBOPBuilder(builders.OfflineBuilder[mbop_networks.MBOPNetworks,
                                          acting.ActorCore, types.Transition]):
  """MBOP Builder.

  This builder uses ensemble regressor learners for the world model, policy
  prior and the n-step return models with fixed learning rates. The ensembles
  and the learning rate are configured in the config.
  """

  def __init__(
      self,
      config: mbop_config.MBOPConfig,
      losses: mbop_losses.MBOPLosses,
      mean_std: Optional[running_statistics.NestedMeanStd] = None,
  ):
    """Initializes an MBOP builder.

    Args:
      config: a config with MBOP hyperparameters.
      losses: MBOP losses.
      mean_std: NestedMeanStd used to normalize the samples.
    """
    self._config = config
    self._losses = losses
    self._mean_std = mean_std

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: mbop_networks.MBOPNetworks,
      dataset: Iterator[types.Transition],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    """See base class."""

    def make_ensemble_regressor_learner(
        name: str,
        logger_fn: loggers.LoggerFactory,
        counter: counting.Counter,
        rng_key: networks_lib.PRNGKey,
        iterator: Iterator[types.Transition],
        network: networks_lib.FeedForwardNetwork,
        loss: mbop_losses.TransitionLoss,
    ) -> core.Learner:
      """Creates an ensemble regressor learner."""
      return learning.make_ensemble_regressor_learner(
          name,
          self._config.num_networks,
          logger_fn,
          counter,
          rng_key,
          iterator,
          network,
          loss,
          optax.adam(self._config.learning_rate),
          self._config.num_sgd_steps_per_step,
      )

    make_world_model_learner = functools.partial(
        make_ensemble_regressor_learner, 'world_model')
    make_policy_prior_learner = functools.partial(
        make_ensemble_regressor_learner, 'policy_prior')
    make_n_step_return_learner = functools.partial(
        make_ensemble_regressor_learner, 'n_step_return')
    counter = counter or counting.Counter(time_delta=0.)
    return learning.MBOPLearner(
        networks,
        self._losses,
        dataset,
        random_key,
        logger_fn,
        make_world_model_learner,
        make_policy_prior_learner,
        make_n_step_return_learner,
        counter,
    )

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy: acting.ActorCore,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
  ) -> core.Actor:
    """See base class."""
    del environment_spec
    return acting.make_actor(policy, random_key, variable_source)

  def make_policy(
      self,
      networks: mbop_networks.MBOPNetworks,
      environment_spec: specs.EnvironmentSpec,
      evaluation: bool,
  ) -> acting.ActorCore:
    """See base class."""
    return acting.make_ensemble_actor_core(
        networks,
        self._config.mppi_config,
        environment_spec,
        self._mean_std,
        use_round_robin=not evaluation)
