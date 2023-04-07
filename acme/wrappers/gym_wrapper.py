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

"""Wraps an OpenAI Gym environment to be used as a dm_env environment."""

from typing import Any, Dict, List, Optional

import dm_env
import gym
import numpy as np
import tree
from gym import spaces

from acme import specs, types


class GymWrapper(dm_env.Environment):
    """Environment wrapper for OpenAI Gym environments."""

    # Note: we don't inherit from base.EnvironmentWrapper because that class
    # assumes that the wrapped environment is a dm_env.Environment.

    def __init__(self, environment: gym.Env):

        self._environment = environment
        self._reset_next_step = True
        self._last_info = None

        # Convert action and observation specs.
        obs_space = self._environment.observation_space
        act_space = self._environment.action_space
        self._observation_spec = _convert_to_spec(obs_space, name="observation")
        self._action_spec = _convert_to_spec(act_space, name="action")

    def reset(self) -> dm_env.TimeStep:
        """Resets the episode."""
        self._reset_next_step = False
        observation = self._environment.reset()
        # Reset the diagnostic information.
        self._last_info = None
        return dm_env.restart(observation)

    def step(self, action: types.NestedArray) -> dm_env.TimeStep:
        """Steps the environment."""
        if self._reset_next_step:
            return self.reset()

        observation, reward, done, info = self._environment.step(action)
        self._reset_next_step = done
        self._last_info = info

        # Convert the type of the reward based on the spec, respecting the scalar or
        # array property.
        reward = tree.map_structure(
            lambda x, t: (  # pylint: disable=g-long-lambda
                t.dtype.type(x) if np.isscalar(x) else np.asarray(x, dtype=t.dtype)
            ),
            reward,
            self.reward_spec(),
        )

        if done:
            truncated = info.get("TimeLimit.truncated", False)
            if truncated:
                return dm_env.truncation(reward, observation)
            return dm_env.termination(reward, observation)
        return dm_env.transition(reward, observation)

    def observation_spec(self) -> types.NestedSpec:
        return self._observation_spec

    def action_spec(self) -> types.NestedSpec:
        return self._action_spec

    def get_info(self) -> Optional[Dict[str, Any]]:
        """Returns the last info returned from env.step(action).

    Returns:
      info: dictionary of diagnostic information from the last environment step
    """
        return self._last_info

    @property
    def environment(self) -> gym.Env:
        """Returns the wrapped environment."""
        return self._environment

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name)
            )
        return getattr(self._environment, name)

    def close(self):
        self._environment.close()


def _convert_to_spec(space: gym.Space, name: Optional[str] = None) -> types.NestedSpec:
    """Converts an OpenAI Gym space to a dm_env spec or nested structure of specs.

  Box, MultiBinary and MultiDiscrete Gym spaces are converted to BoundedArray
  specs. Discrete OpenAI spaces are converted to DiscreteArray specs. Tuple and
  Dict spaces are recursively converted to tuples and dictionaries of specs.

  Args:
    space: The Gym space to convert.
    name: Optional name to apply to all return spec(s).

  Returns:
    A dm_env spec or nested structure of specs, corresponding to the input
    space.
  """
    if isinstance(space, spaces.Discrete):
        return specs.DiscreteArray(num_values=space.n, dtype=space.dtype, name=name)

    elif isinstance(space, spaces.Box):
        return specs.BoundedArray(
            shape=space.shape,
            dtype=space.dtype,
            minimum=space.low,
            maximum=space.high,
            name=name,
        )

    elif isinstance(space, spaces.MultiBinary):
        return specs.BoundedArray(
            shape=space.shape, dtype=space.dtype, minimum=0.0, maximum=1.0, name=name
        )

    elif isinstance(space, spaces.MultiDiscrete):
        return specs.BoundedArray(
            shape=space.shape,
            dtype=space.dtype,
            minimum=np.zeros(space.shape),
            maximum=space.nvec - 1,
            name=name,
        )

    elif isinstance(space, spaces.Tuple):
        return tuple(_convert_to_spec(s, name) for s in space.spaces)

    elif isinstance(space, spaces.Dict):
        return {
            key: _convert_to_spec(value, key) for key, value in space.spaces.items()
        }

    else:
        raise ValueError("Unexpected gym space: {}".format(space))


class GymAtariAdapter(GymWrapper):
    """Specialized wrapper exposing a Gym Atari environment.

  This wraps the Gym Atari environment in the same way as GymWrapper, but also
  exposes the lives count as an observation. The resuling observations are
  a tuple whose first element is the RGB observations and the second is the
  lives count.
  """

    def _wrap_observation(self, observation: types.NestedArray) -> types.NestedArray:
        # pytype: disable=attribute-error
        return observation, self._environment.ale.lives()
        # pytype: enable=attribute-error

    def reset(self) -> dm_env.TimeStep:
        """Resets the episode."""
        self._reset_next_step = False
        observation = self._environment.reset()
        observation = self._wrap_observation(observation)
        return dm_env.restart(observation)

    def step(self, action: List[np.ndarray]) -> dm_env.TimeStep:
        """Steps the environment."""
        if self._reset_next_step:
            return self.reset()

        observation, reward, done, _ = self._environment.step(action[0].item())
        self._reset_next_step = done

        observation = self._wrap_observation(observation)

        if done:
            return dm_env.termination(reward, observation)
        return dm_env.transition(reward, observation)

    def observation_spec(self) -> types.NestedSpec:
        return (
            self._observation_spec,
            specs.Array(shape=(), dtype=np.dtype("float64"), name="lives"),
        )

    def action_spec(self) -> List[specs.BoundedArray]:
        return [self._action_spec]  # pytype: disable=bad-return-type
