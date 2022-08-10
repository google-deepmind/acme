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

"""BC Builder."""
from typing import Iterator, Optional

from acme import core
from acme import specs
from acme import types
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.agents.jax.bc import config as bc_config
from acme.agents.jax.bc import learning
from acme.agents.jax.bc import losses
from acme.agents.jax.bc import networks as bc_networks
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import jax
import optax


class BCBuilder(builders.OfflineBuilder[bc_networks.BCNetworks,
                                        actor_core_lib.FeedForwardPolicy,
                                        types.Transition]):
  """BC Builder."""

  def __init__(
      self,
      config: bc_config.BCConfig,
      loss_fn: losses.BCLoss,
      loss_has_aux: bool = False,
  ):
    """Creates a BC learner, an evaluation policy and an eval actor.

    Args:
      config: a config with BC hps.
      loss_fn: BC loss to use.
      loss_has_aux: Whether the loss function returns auxiliary metrics as a
        second argument.
    """
    self._config = config
    self._loss_fn = loss_fn
    self._loss_has_aux = loss_has_aux

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: bc_networks.BCNetworks,
      dataset: Iterator[types.Transition],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      *,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    del environment_spec

    return learning.BCLearner(
        networks=networks,
        random_key=random_key,
        loss_fn=self._loss_fn,
        optimizer=optax.adam(learning_rate=self._config.learning_rate),
        prefetching_iterator=utils.sharded_prefetch(dataset),
        num_sgd_steps_per_step=self._config.num_sgd_steps_per_step,
        loss_has_aux=self._loss_has_aux,
        logger=logger_fn('learner'),
        counter=counter)

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy: actor_core_lib.FeedForwardPolicy,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
  ) -> core.Actor:
    del environment_spec
    assert variable_source is not None
    actor_core = actor_core_lib.batched_feed_forward_to_actor_core(policy)
    variable_client = variable_utils.VariableClient(
        variable_source, 'policy', device='cpu')
    return actors.GenericActor(
        actor_core, random_key, variable_client, backend='cpu')

  def make_policy(self,
                  networks: bc_networks.BCNetworks,
                  environment_spec: specs.EnvironmentSpec,
                  evaluation: bool = False) -> actor_core_lib.FeedForwardPolicy:
    """Construct the policy."""
    del environment_spec, evaluation

    def evaluation_policy(
        params: networks_lib.Params, key: networks_lib.PRNGKey,
        observation: networks_lib.Observation) -> networks_lib.Action:
      apply_key, sample_key = jax.random.split(key)
      network_output = networks.policy_network.apply(
          params, observation, is_training=False, key=apply_key)
      return networks.sample_fn(network_output, sample_key)

    return evaluation_policy
