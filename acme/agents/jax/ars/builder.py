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

"""ARS Builder."""
from typing import Dict, Iterator, List, Optional, Tuple

import acme
from acme import adders
from acme import core
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.agents.jax.ars import config as ars_config
from acme.agents.jax.ars import learning
from acme.jax import networks as networks_lib
from acme.jax import running_statistics
from acme.jax import utils
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import numpy as np
import reverb


def get_policy(policy_network: networks_lib.FeedForwardNetwork,
               normalization_apply_fn) -> actor_core_lib.FeedForwardPolicy:
  """Returns a function that computes actions."""

  def apply(
      params: networks_lib.Params, key: networks_lib.PRNGKey,
      obs: networks_lib.Observation
  ) -> Tuple[networks_lib.Action, Dict[str, jnp.ndarray]]:
    del key
    params_key, policy_params, normalization_params = params
    normalized_obs = normalization_apply_fn(obs, normalization_params)
    action = policy_network.apply(policy_params, normalized_obs)
    return action, {
        'params_key':
            jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), params_key)
    }

  return apply


class ARSBuilder(
    builders.ActorLearnerBuilder[networks_lib.FeedForwardNetwork,
                                 Tuple[str, networks_lib.FeedForwardNetwork],
                                 reverb.ReplaySample]):
  """ARS Builder."""

  def __init__(
      self,
      config: ars_config.ARSConfig,
      spec: specs.EnvironmentSpec,
  ):
    self._config = config
    self._spec = spec

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: networks_lib.FeedForwardNetwork,
      dataset: Iterator[reverb.ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    del environment_spec, replay_client
    return learning.ARSLearner(self._spec, networks, random_key, self._config,
                               dataset, counter, logger_fn('learner'))

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy: Tuple[str, networks_lib.FeedForwardNetwork],
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
      adder: Optional[adders.Adder] = None,
  ) -> acme.Actor:
    del environment_spec
    assert variable_source is not None

    kname, policy = policy

    normalization_apply_fn = (
        running_statistics.normalize if self._config.normalize_observations else
        (lambda a, b: a))
    policy_to_run = get_policy(policy, normalization_apply_fn)

    actor_core = actor_core_lib.batched_feed_forward_with_extras_to_actor_core(
        policy_to_run)
    variable_client = variable_utils.VariableClient(variable_source, kname,
                                                    device='cpu')
    return actors.GenericActor(
        actor_core,
        random_key,
        variable_client,
        adder,
        backend='cpu',
        per_episode_update=True)

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy: Tuple[str, networks_lib.FeedForwardNetwork],
  ) -> List[reverb.Table]:
    """Create tables to insert data into."""
    del policy
    extra_spec = {
        'params_key': (np.zeros(shape=(), dtype=np.int32),
                       np.zeros(shape=(), dtype=np.int32),
                       np.zeros(shape=(), dtype=np.bool_)),
    }
    signature = adders_reverb.EpisodeAdder.signature(
        environment_spec, sequence_length=None, extras_spec=extra_spec)
    return [
        reverb.Table.queue(
            name=self._config.replay_table_name,
            max_size=10000,  # a big number
            signature=signature)
    ]

  def make_dataset_iterator(
      self, replay_client: reverb.Client) -> Iterator[reverb.ReplaySample]:
    """Create a dataset iterator to use for learning/updating the agent."""
    dataset = reverb.TrajectoryDataset.from_table_signature(
        server_address=replay_client.server_address,
        table=self._config.replay_table_name,
        max_in_flight_samples_per_worker=1)
    return utils.device_put(dataset.as_numpy_iterator(), jax.devices()[0])

  def make_adder(self,
                 replay_client: reverb.Client) -> Optional[adders.Adder]:
    """Create an adder which records data generated by the actor/environment."""

    return adders_reverb.EpisodeAdder(
        priority_fns={self._config.replay_table_name: None},
        client=replay_client,
        max_sequence_length=2000,
    )
