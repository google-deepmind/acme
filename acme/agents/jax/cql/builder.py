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

"""CQL Builder."""
from typing import Iterator, Optional

from acme import core
from acme import specs
from acme import types
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.agents.jax.cql import config as cql_config
from acme.agents.jax.cql import learning
from acme.agents.jax.cql import networks as cql_networks
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import optax


class CQLBuilder(builders.OfflineBuilder[cql_networks.CQLNetworks,
                                         actor_core_lib.FeedForwardPolicy,
                                         types.Transition]):
  """CQL Builder."""

  def __init__(
      self,
      config: cql_config.CQLConfig,
  ):
    """Creates a CQL learner, an evaluation policy and an eval actor.

    Args:
      config: a config with CQL hps.
    """
    self._config = config

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: cql_networks.CQLNetworks,
      dataset: Iterator[types.Transition],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      *,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    del environment_spec

    return learning.CQLLearner(
        batch_size=self._config.batch_size,
        networks=networks,
        random_key=random_key,
        demonstrations=dataset,
        policy_optimizer=optax.adam(self._config.policy_learning_rate),
        critic_optimizer=optax.adam(self._config.critic_learning_rate),
        tau=self._config.tau,
        fixed_cql_coefficient=self._config.fixed_cql_coefficient,
        cql_lagrange_threshold=self._config.cql_lagrange_threshold,
        cql_num_samples=self._config.cql_num_samples,
        num_sgd_steps_per_step=self._config.num_sgd_steps_per_step,
        reward_scale=self._config.reward_scale,
        discount=self._config.discount,
        fixed_entropy_coefficient=self._config.fixed_entropy_coefficient,
        target_entropy=self._config.target_entropy,
        num_bc_iters=self._config.num_bc_iters,
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

  def make_policy(self, networks: cql_networks.CQLNetworks,
                  environment_spec: specs.EnvironmentSpec,
                  evaluation: bool) -> actor_core_lib.FeedForwardPolicy:
    """Construct the policy."""
    del environment_spec, evaluation

    def evaluation_policy(
        params: networks_lib.Params, key: networks_lib.PRNGKey,
        observation: networks_lib.Observation) -> networks_lib.Action:
      dist_params = networks.policy_network.apply(params, observation)
      return networks.sample_eval(dist_params, key)

    return evaluation_policy
