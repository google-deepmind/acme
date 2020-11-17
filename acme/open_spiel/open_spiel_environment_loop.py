# python3
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

"""An OpenSpiel agent-environment training loop."""

import operator
import time
from typing import List, Optional

from acme import core
# Internal imports.
from acme.utils import counting
from acme.utils import loggers

import dm_env
from dm_env import specs
import numpy as np
import tree


class OpenSpielEnvironmentLoop(core.Worker):
  """A simple RL environment loop.

  This takes `Environment` and `Actor` instances and coordinates their
  interaction. This can be used as:

    loop = EnvironmentLoop(environment, actor)
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
      environment: dm_env.Environment,
      actors: List[core.Actor],
      counter: counting.Counter = None,
      logger: loggers.Logger = None,
      label: str = 'open_spiel_environment_loop',
  ):
    # Internalize agent and environment.
    self._environment = environment
    self._actors = actors
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(label)

  def run_episode(self, verbose=False) -> loggers.LoggingData:
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
        (self._environment._game.num_players(),),
        np.float32,
        minimum=self._environment._game.min_utility(),
        maximum=self._environment._game.max_utility())
    episode_return = tree.map_structure(_generate_zeros_from_spec,
                                        multiplayer_reward_spec)

    timestep = self._environment.reset()

    # Make the first observation.
    # TODO Note: OpenSpiel agents handle observe_first() internally.
    for actor in self._actors:
      actor.observe(None, next_timestep=timestep)

    # Run an episode.
    while not timestep.last():
      # Generate an action from the agent's policy and step the environment.
      pid = timestep.observation["current_player"]

      if self._environment.is_turn_based:
        action_list = [self._actors[pid].select_action(timestep.observation)]
      else:
        # TODO Test this on simultaneous move games.
        agents_output = [agent.step(time_step) for agent in agents]
        action_list = [
            actor.select_action(timestep.observation) for actor in self._actors
        ]

      # TODO Delete or move to logger?
      if verbose:
        self._actors[pid].print_policy(timestep.observation)
        print("Action: ", action_list[0])

      timestep = self._environment.step(action_list)

      # TODO Delete or move to logger?
      if verbose:
        print("State:")
        print(str(self._environment._state))

      # Have the agent observe the timestep and let the actor update itself.
      for actor in self._actors:
        actor.observe(action_list, next_timestep=timestep)

      # Book-keeping.
      episode_steps += 1

      # Equivalent to: episode_return += timestep.reward
      tree.map_structure(operator.iadd, episode_return, timestep.reward)

      # TODO Delete or move to logger?
      if verbose:
        print("Reward: ", timestep.reward)

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
      # TODO Remove verbose?
      if episode_count % 1000 == 0:
        result = self.run_episode(verbose=True)
      else:
        result = self.run_episode(verbose=False)
      episode_count += 1
      step_count += result['episode_length']
      # Log the given results.
      self._logger.write(result)


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
  return np.zeros(spec.shape, spec.dtype)


# Internal class.
