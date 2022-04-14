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

"""An OpenSpiel multi-agent/environment training loop."""

import operator
import time
from typing import Optional, Sequence

from acme import core
from acme.utils import counting
from acme.utils import loggers
from acme.wrappers import open_spiel_wrapper
import dm_env
from dm_env import specs
import numpy as np
import tree

# pytype: disable=import-error
import pyspiel
# pytype: enable=import-error


class OpenSpielEnvironmentLoop(core.Worker):
  """An OpenSpiel RL environment loop.

  This takes `Environment` and list of `Actor` instances and coordinates their
  interaction. Agents are updated if `should_update=True`. This can be used as:

    loop = EnvironmentLoop(environment, actors)
    loop.run(num_episodes)

  A `Counter` instance can optionally be given in order to maintain counts
  between different Acme components. If not given a local Counter will be
  created to maintain counts between calls to the `run` method.

  A `Logger` instance can also be passed in order to control the output of the
  loop. If not given a platform-specific default logger will be used as defined
  by utils.loggers.make_default_logger. A string `label` can be passed to easily
  change the label associated with the default logger; this is ignored if a
  `Logger` instance is given.
  """

  def __init__(
      self,
      environment: open_spiel_wrapper.OpenSpielWrapper,
      actors: Sequence[core.Actor],
      counter: Optional[counting.Counter] = None,
      logger: Optional[loggers.Logger] = None,
      should_update: bool = True,
      label: str = 'open_spiel_environment_loop',
  ):
    # Internalize agent and environment.
    self._environment = environment
    self._actors = actors
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(label)
    self._should_update = should_update

    # Track information necessary to coordinate updates among multiple actors.
    self._observed_first = [False] * len(self._actors)
    self._prev_actions = [pyspiel.INVALID_ACTION] * len(self._actors)

  def _send_observation(self, timestep: dm_env.TimeStep, player: int):
    # If terminal all actors must update
    if player == pyspiel.PlayerId.TERMINAL:
      for player_id in range(len(self._actors)):
        # Note: we must account for situations where the first observation
        # is a terminal state, e.g. if an opponent folds in poker before we get
        # to act.
        if self._observed_first[player_id]:
          player_timestep = self._get_player_timestep(timestep, player_id)
          self._actors[player_id].observe(self._prev_actions[player_id],
                                          player_timestep)
          if self._should_update:
            self._actors[player_id].update()
      self._observed_first = [False] * len(self._actors)
      self._prev_actions = [pyspiel.INVALID_ACTION] * len(self._actors)
    else:
      if not self._observed_first[player]:
        player_timestep = dm_env.TimeStep(
            observation=timestep.observation[player],
            reward=None,
            discount=None,
            step_type=dm_env.StepType.FIRST)
        self._actors[player].observe_first(player_timestep)
        self._observed_first[player] = True
      else:
        player_timestep = self._get_player_timestep(timestep, player)
        self._actors[player].observe(self._prev_actions[player],
                                     player_timestep)
        if self._should_update:
          self._actors[player].update()

  def _get_action(self, timestep: dm_env.TimeStep, player: int) -> int:
    self._prev_actions[player] = self._actors[player].select_action(
        timestep.observation[player])
    return self._prev_actions[player]

  def _get_player_timestep(self, timestep: dm_env.TimeStep,
                           player: int) -> dm_env.TimeStep:
    return dm_env.TimeStep(observation=timestep.observation[player],
                           reward=timestep.reward[player],
                           discount=timestep.discount[player],
                           step_type=timestep.step_type)

  def run_episode(self) -> loggers.LoggingData:
    """Run one episode.

    Each episode is a loop which interacts first with the environment to get an
    observation and then give that observation to the agent in order to retrieve
    an action.

    Returns:
      An instance of `loggers.LoggingData`.
    """
    # Reset any counts and start the environment.
    start_time = time.time()
    episode_steps = 0

    # For evaluation, this keeps track of the total undiscounted reward
    # for each player accumulated during the episode.
    multiplayer_reward_spec = specs.BoundedArray(
        (self._environment.game.num_players(),),
        np.float32,
        minimum=self._environment.game.min_utility(),
        maximum=self._environment.game.max_utility())
    episode_return = tree.map_structure(_generate_zeros_from_spec,
                                        multiplayer_reward_spec)

    timestep = self._environment.reset()

    # Make the first observation.
    self._send_observation(timestep, self._environment.current_player)

    # Run an episode.
    while not timestep.last():
      # Generate an action from the agent's policy and step the environment.
      if self._environment.is_turn_based:
        action_list = [
            self._get_action(timestep, self._environment.current_player)
        ]
      else:
        # FIXME: Support simultaneous move games.
        raise ValueError('Currently only supports sequential games.')

      timestep = self._environment.step(action_list)

      # Have the agent observe the timestep and let the actor update itself.
      self._send_observation(timestep, self._environment.current_player)

      # Book-keeping.
      episode_steps += 1

      # Equivalent to: episode_return += timestep.reward
      # We capture the return value because if timestep.reward is a JAX
      # DeviceArray, episode_return will not be mutated in-place. (In all other
      # cases, the returned episode_return will be the same object as the
      # argument episode_return.)
      episode_return = tree.map_structure(operator.iadd,
                                          episode_return,
                                          timestep.reward)

    # Record counts.
    counts = self._counter.increment(episodes=1, steps=episode_steps)

    # Collect the results and combine with counts.
    steps_per_second = episode_steps / (time.time() - start_time)
    result = {
        'episode_length': episode_steps,
        'episode_return': episode_return,
        'steps_per_second': steps_per_second,
    }
    result.update(counts)
    return result

  def run(self,
          num_episodes: Optional[int] = None,
          num_steps: Optional[int] = None):
    """Perform the run loop.

    Run the environment loop either for `num_episodes` episodes or for at
    least `num_steps` steps (the last episode is always run until completion,
    so the total number of steps may be slightly more than `num_steps`).
    At least one of these two arguments has to be None.

    Upon termination of an episode a new episode will be started. If the number
    of episodes and the number of steps are not given then this will interact
    with the environment infinitely.

    Args:
      num_episodes: number of episodes to run the loop for.
      num_steps: minimal number of steps to run the loop for.

    Raises:
      ValueError: If both 'num_episodes' and 'num_steps' are not None.
    """

    if not (num_episodes is None or num_steps is None):
      raise ValueError('Either "num_episodes" or "num_steps" should be None.')

    def should_terminate(episode_count: int, step_count: int) -> bool:
      return ((num_episodes is not None and episode_count >= num_episodes) or
              (num_steps is not None and step_count >= num_steps))

    episode_count, step_count = 0, 0
    while not should_terminate(episode_count, step_count):
      result = self.run_episode()
      episode_count += 1
      step_count += result['episode_length']
      # Log the given results.
      self._logger.write(result)


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)
