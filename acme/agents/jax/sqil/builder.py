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

"""SQIL Builder (https://arxiv.org/pdf/1905.11108.pdf)."""

from typing import Callable, Generic, Iterator, List, Optional

from acme import adders
from acme import core
from acme import specs
from acme import types
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax.imitation_learning_types import DirectPolicyNetwork, DirectRLNetworks  # pylint: disable=g-multiple-import
from acme.utils import counting
from acme.utils import loggers
import jax
import numpy as np
import reverb
import tree


def _generate_sqil_samples(
    demonstration_iterator: Iterator[types.Transition],
    replay_iterator: Iterator[reverb.ReplaySample]
) -> Iterator[reverb.ReplaySample]:
  """Generator which creates the sample iterator for SQIL.

  Args:
    demonstration_iterator: Iterator of demonstrations.
    replay_iterator: Replay buffer sample iterator.

  Yields:
    Samples having a mix of demonstrations with reward 1 and replay samples with
    reward 0.
  """
  for demonstrations, replay_sample in zip(demonstration_iterator,
                                           replay_iterator):
    demonstrations = demonstrations._replace(
        reward=np.ones_like(demonstrations.reward))

    replay_transitions = replay_sample.data
    replay_transitions = replay_transitions._replace(
        reward=np.zeros_like(replay_transitions.reward))

    double_batch = tree.map_structure(lambda x, y: np.concatenate([x, y]),
                                      demonstrations, replay_transitions)

    # Split the double batch in an interleaving fashion.
    # e.g [1, 2, 3, 4 ,5 ,6] -> [1, 3, 5] and [2, 4, 6]
    yield reverb.ReplaySample(
        info=replay_sample.info,
        data=tree.map_structure(lambda x: x[0::2], double_batch))
    yield reverb.ReplaySample(
        info=replay_sample.info,
        data=tree.map_structure(lambda x: x[1::2], double_batch))


class SQILBuilder(Generic[DirectRLNetworks, DirectPolicyNetwork],
                  builders.ActorLearnerBuilder[DirectRLNetworks,
                                               DirectPolicyNetwork,
                                               reverb.ReplaySample]):
  """SQIL Builder (https://openreview.net/pdf?id=S1xKd24twB)."""

  def __init__(self,
               rl_agent: builders.ActorLearnerBuilder[DirectRLNetworks,
                                                      DirectPolicyNetwork,
                                                      reverb.ReplaySample],
               rl_agent_batch_size: int,
               make_demonstrations: Callable[[int],
                                             Iterator[types.Transition]]):
    """Builds a SQIL agent.

    Args:
      rl_agent: An off policy direct RL agent..
      rl_agent_batch_size: The batch size of the above algorithm.
      make_demonstrations: A function that returns an infinite iterator with
        demonstrations.
    """
    self._rl_agent = rl_agent
    self._rl_agent_batch_size = rl_agent_batch_size
    self._make_demonstrations = make_demonstrations

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: DirectRLNetworks,
      dataset: Iterator[reverb.ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: Optional[specs.EnvironmentSpec] = None,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    """Creates the learner."""
    counter = counter or counting.Counter()
    direct_rl_counter = counting.Counter(counter, 'direct_rl')
    return self._rl_agent.make_learner(
        random_key,
        networks,
        dataset=dataset,
        logger_fn=logger_fn,
        environment_spec=environment_spec,
        replay_client=replay_client,
        counter=direct_rl_counter)

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy: DirectPolicyNetwork,
  ) -> List[reverb.Table]:
    return self._rl_agent.make_replay_tables(environment_spec, policy)

  def make_dataset_iterator(
      self,
      replay_client: reverb.Client) -> Optional[Iterator[reverb.ReplaySample]]:
    """The returned iterator returns batches with both expert and policy data.

    Batch items will alternate between expert data and policy data.

    Args:
      replay_client: Reverb client.

    Returns:
      The Replay sample iterator.
    """
    # TODO(eorsini): Make sure we have the exact same format as the rl_agent's
    # adder writes in.
    demonstration_iterator = self._make_demonstrations(
        self._rl_agent_batch_size)

    rb_iterator = self._rl_agent.make_dataset_iterator(replay_client)

    return utils.device_put(
        _generate_sqil_samples(demonstration_iterator, rb_iterator),
        jax.devices()[0])

  def make_adder(self,
                 replay_client: reverb.Client) -> Optional[adders.Adder]:
    return self._rl_agent.make_adder(replay_client)

  def make_actor(
      self,
      random_key: networks_lib.PRNGKey,
      policy: DirectPolicyNetwork,
      environment_spec: specs.EnvironmentSpec,
      variable_source: Optional[core.VariableSource] = None,
      adder: Optional[adders.Adder] = None,
  ) -> core.Actor:
    return self._rl_agent.make_actor(random_key, policy, environment_spec,
                                     variable_source, adder)

  def make_policy(self,
                  networks: DirectRLNetworks,
                  environment_spec: specs.EnvironmentSpec,
                  evaluation: bool = False) -> DirectPolicyNetwork:
    return self._rl_agent.make_policy(networks, environment_spec, evaluation)
