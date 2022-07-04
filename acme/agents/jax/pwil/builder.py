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

"""PWIL agent implementation, using JAX."""

import threading
from typing import Callable, Generic, Iterator, List, Optional, Sequence

from acme import adders
from acme import core
from acme import specs
from acme import types
from acme.agents.jax import builders
from acme.agents.jax.pwil import adder as pwil_adder
from acme.agents.jax.pwil import config as pwil_config
from acme.agents.jax.pwil import rewarder
from acme.jax import networks as networks_lib
from acme.jax.imitation_learning_types import DirectPolicyNetwork, DirectRLNetworks  # pylint: disable=g-multiple-import
from acme.jax.types import PRNGKey
from acme.utils import counting
from acme.utils import loggers
import dm_env
import numpy as np
import reverb


def _prefill_with_demonstrations(adder: adders.Adder,
                                 demonstrations: Sequence[types.Transition],
                                 reward: Optional[float],
                                 min_num_transitions: int = 0) -> None:
  """Fill the adder's replay buffer with expert transitions.

  Assumes that the demonstrations dataset stores transitions in order.

  Args:
    adder: the agent which adds the demonstrations.
    demonstrations: the expert demonstrations to iterate over.
    reward: if non-None, populates the environment reward entry of transitions.
    min_num_transitions: the lower bound on transitions processed, the dataset
      will be iterated over multiple times if needed. Once at least
      min_num_transitions are added, the processing is interrupted at the
      nearest episode end.
  """
  if not demonstrations:
    return

  reward = np.float32(reward) if reward is not None else reward
  remaining_transitions = min_num_transitions
  step_type = None
  action = None
  ts = dm_env.TimeStep(None, None, None, None)  # Unused.
  while remaining_transitions > 0:
    # In case we share the adder or demonstrations don't end with
    # end-of-episode, reset the adder prior to add_first.
    adder.reset()
    for transition_num, transition in enumerate(demonstrations):
      remaining_transitions -= 1
      discount = np.float32(1.0)
      ts_reward = reward if reward is not None else transition.reward
      if step_type == dm_env.StepType.LAST or transition_num == 0:
        ts = dm_env.TimeStep(dm_env.StepType.FIRST, ts_reward, discount,
                             transition.observation)
        adder.add_first(ts)

      observation = transition.next_observation
      action = transition.action
      if transition.discount == 0. or transition_num == len(demonstrations) - 1:
        step_type = dm_env.StepType.LAST
        discount = np.float32(0.0)
      else:
        step_type = dm_env.StepType.MID
      ts = dm_env.TimeStep(step_type, ts_reward, discount, observation)
      adder.add(action, ts)
      if remaining_transitions <= 0:
        # Note: we could check `step_type == dm_env.StepType.LAST` to stop at an
        # episode end if possible.
        break

  # Explicitly finalize the Reverb client writes.
  adder.reset()


class PWILBuilder(builders.ActorLearnerBuilder[DirectRLNetworks,
                                               DirectPolicyNetwork,
                                               reverb.ReplaySample],
                  Generic[DirectRLNetworks, DirectPolicyNetwork]):
  """PWIL Agent builder."""

  def __init__(self,
               rl_agent: builders.ActorLearnerBuilder[DirectRLNetworks,
                                                      DirectPolicyNetwork,
                                                      reverb.ReplaySample],
               config: pwil_config.PWILConfig,
               demonstrations_fn: Callable[[], pwil_config.PWILDemonstrations]):
    """Initialize the agent.

    Args:
      rl_agent: the standard RL algorithm.
      config: PWIL-specific configuration.
      demonstrations_fn: A function that returns an iterator over contiguous
        demonstration transitions, and the average demonstration episode length.
    """
    self._rl_agent = rl_agent
    self._config = config
    self._demonstrations_fn = demonstrations_fn
    super().__init__()

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: DirectRLNetworks,
      dataset: Iterator[reverb.ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: specs.EnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    return self._rl_agent.make_learner(
        random_key=random_key,
        networks=networks,
        dataset=dataset,
        logger_fn=logger_fn,
        environment_spec=environment_spec,
        replay_client=replay_client,
        counter=counter)

  def make_replay_tables(
      self,
      environment_spec: specs.EnvironmentSpec,
      policy: DirectPolicyNetwork,
  ) -> List[reverb.Table]:
    return self._rl_agent.make_replay_tables(environment_spec, policy)

  def make_dataset_iterator(
      self,
      replay_client: reverb.Client) -> Optional[Iterator[reverb.ReplaySample]]:
    # make_dataset_iterator is only called once (per learner), to pass the
    # iterator to make_learner. By using adders we ensure the transition types
    # (e.g. n-step transitions) that the direct RL agent expects.
    if self._config.num_transitions_rb > 0:

      def prefill_thread():
        # Populating the replay buffer with the direct RL agent guarantees that
        # a constant reward will be used, not the imitation reward.
        prefill_reward = (
            self._config.alpha
            if self._config.prefill_constant_reward else None)
        _prefill_with_demonstrations(
            adder=self._rl_agent.make_adder(replay_client),
            demonstrations=list(self._demonstrations_fn().demonstrations),
            min_num_transitions=self._config.num_transitions_rb,
            reward=prefill_reward)
      # Populate the replay buffer in a separate thread, so that the learner
      # can sample from the buffer, to avoid blocking on the buffer being full.
      threading.Thread(target=prefill_thread, daemon=True).start()

    return self._rl_agent.make_dataset_iterator(replay_client)

  def make_adder(self, replay_client: reverb.Client) -> adders.Adder:
    """Creates the adder substituting imitation reward."""
    pwil_demonstrations = self._demonstrations_fn()
    return pwil_adder.PWILAdder(
        direct_rl_adder=self._rl_agent.make_adder(replay_client),
        pwil_rewarder=rewarder.WassersteinDistanceRewarder(
            demonstrations_it=pwil_demonstrations.demonstrations,
            episode_length=pwil_demonstrations.episode_length,
            use_actions_for_distance=self._config.use_actions_for_distance,
            alpha=self._config.alpha,
            beta=self._config.beta))

  def make_actor(
      self,
      random_key: PRNGKey,
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
