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

"""IQL Builder."""
from typing import Iterator, Optional

from acme import core
from acme import specs
from acme import types
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.agents.jax.iql import config as iql_config
from acme.agents.jax.iql import learning
from acme.agents.jax.iql import networks as iql_networks
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import optax


class IQLBuilder(builders.OfflineBuilder[iql_networks.IQLNetworks,
                                         actor_core_lib.FeedForwardPolicy,
                                         types.Transition]):
  """IQL Builder.
  
  Constructs the components needed for Implicit Q-Learning agent,
  including the learner, policy, and actor.
  """

  def __init__(self, config: iql_config.IQLConfig):
    """Creates an IQL builder.

    Args:
      config: Configuration with IQL hyperparameters.
    """
    self._config = config

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: iql_networks.IQLNetworks,
      dataset: Iterator[types.Transition],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      *,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    """Creates an IQL learner.
    
    Args:
      random_key: Random number generator key.
      networks: IQL networks (policy, Q-function, value function).
      dataset: Iterator over offline dataset.
      logger_fn: Factory for creating loggers.
      environment_spec: Environment specification.
      counter: Counter for tracking training progress.
    
    Returns:
      IQL learner instance.
    """
    del environment_spec

    return learning.IQLLearner(
        batch_size=self._config.batch_size,
        networks=networks,
        random_key=random_key,
        demonstrations=dataset,
        policy_optimizer=optax.adam(self._config.policy_learning_rate),
        value_optimizer=optax.adam(self._config.value_learning_rate),
        critic_optimizer=optax.adam(self._config.critic_learning_rate),
        tau=self._config.tau,
        expectile=self._config.expectile,
        temperature=self._config.temperature,
        discount=self._config.discount,
        num_sgd_steps_per_step=self._config.num_sgd_steps_per_step,
        logger=logger_fn('learner'),
        counter=counter)

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy: actor_core_lib.FeedForwardPolicy,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
  ) -> core.Actor:
    """Creates an actor for policy evaluation.
    
    Args:
      random_key: Random number generator key.
      policy: Policy function to execute.
      environment_spec: Environment specification.
      variable_source: Source for policy parameters.
    
    Returns:
      Actor instance.
    """
    del environment_spec
    assert variable_source is not None
    actor_core = actor_core_lib.batched_feed_forward_to_actor_core(policy)
    variable_client = variable_utils.VariableClient(
        variable_source, 'policy', device='cpu')
    return actors.GenericActor(
        actor_core, random_key, variable_client, backend='cpu')

  def make_policy(
      self,
      networks: iql_networks.IQLNetworks,
      environment_spec: specs.EnvironmentSpec,
      evaluation: bool) -> actor_core_lib.FeedForwardPolicy:
    """Constructs the policy function.
    
    Args:
      networks: IQL networks.
      environment_spec: Environment specification.
      evaluation: Whether this is for evaluation (deterministic) or training.
    
    Returns:
      Policy function that maps (params, key, observation) -> action.
    """
    del environment_spec, evaluation

    def policy(
        params: networks_lib.Params,
        key: networks_lib.PRNGKey,
        observation: networks_lib.Observation) -> networks_lib.Action:
      """Evaluation policy (deterministic)."""
      dist_params = networks.policy_network.apply(params, observation)
      return networks.sample_eval(dist_params, key)

    return policy
