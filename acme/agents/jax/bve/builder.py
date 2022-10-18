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

"""BVE Builder."""
from typing import Iterator, Optional

from acme import core
from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.agents.jax.bve import losses
from acme.agents.jax.bve import networks as bve_networks
from acme.agents.jax.dqn import learning_lib
from acme.jax import networks as networks_lib
from acme.jax import types as jax_types
from acme.jax import utils
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import haiku as hk
import optax


class BVEBuilder(builders.OfflineBuilder[bve_networks.BVENetworks,
                                         actor_core_lib.ActorCore,
                                         utils.PrefetchingSplit]):
  """BVE Builder."""

  def __init__(self, config):
    """Build a BVE agent.

    Args:
      config: The config of the BVE agent.
    """
    self._config = config

  def make_learner(self,
                   random_key: jax_types.PRNGKey,
                   networks: bve_networks.BVENetworks,
                   dataset: Iterator[utils.PrefetchingSplit],
                   logger_fn: loggers.LoggerFactory,
                   environment_spec: specs.EnvironmentSpec,
                   counter: Optional[counting.Counter] = None) -> core.Learner:
    del environment_spec

    loss_fn = losses.BVELoss(
        discount=self._config.discount,
        max_abs_reward=self._config.max_abs_reward,
        huber_loss_parameter=self._config.huber_loss_parameter,
    )

    return learning_lib.SGDLearner(
        network=networks.policy_network,
        random_key=random_key,
        optimizer=optax.adam(
            self._config.learning_rate, eps=self._config.adam_eps),
        target_update_period=self._config.target_update_period,
        data_iterator=dataset,
        loss_fn=loss_fn,
        counter=counter,
        num_sgd_steps_per_step=self._config.num_sgd_steps_per_step,
        logger=logger_fn('learner'))

  def make_actor(
      self,
      random_key: jax_types.PRNGKey,
      policy: actor_core_lib.ActorCore,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None) -> core.Actor:
    """Create the actor for the BVE to perform online evals.

    Args:
      random_key: prng key.
      policy: The DQN policy.
      environment_spec: The environment spec.
      variable_source: The source of where the variables are coming from.

    Returns:
      Return the actor for the evaluations.
    """
    del environment_spec
    variable_client = variable_utils.VariableClient(
        variable_source, 'policy', device='cpu')
    return actors.GenericActor(policy, random_key, variable_client)

  def make_policy(
      self,
      networks: bve_networks.BVENetworks,
      environment_spec: specs.EnvironmentSpec,
      evaluation: Optional[bool] = False) -> actor_core_lib.ActorCore:
    """Creates a policy."""
    del environment_spec, evaluation

    def behavior_policy(
        params: hk.Params, key: jax_types.PRNGKey,
        observation: networks_lib.Observation) -> networks_lib.Action:
      network_output = networks.policy_network.apply(
          params, observation, is_training=False)
      return networks.sample_fn(network_output, key)

    return actor_core_lib.batched_feed_forward_to_actor_core(behavior_policy)
