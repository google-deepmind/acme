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

"""Wraps a Multigrid multiagent environment to be used as a dm_env."""

from typing import Any, Dict, List, Optional
import warnings

from acme import specs
from acme import types
from acme import wrappers
from acme.multiagent import types as ma_types
from acme.wrappers import multiagent_dict_key_wrapper
import dm_env
import gym
from gym import spaces
import jax
import numpy as np
import tree

try:
  # The following import registers multigrid environments in gym. Do not remove.
  # pylint: disable=unused-import, disable=g-import-not-at-top
  # pytype: disable=import-error
  from social_rl.gym_multigrid import multigrid
  # pytype: enable=import-error
  # pylint: enable=unused-import, enable=g-import-not-at-top
except ModuleNotFoundError as err:
  raise ModuleNotFoundError(
      'The multiagent multigrid environment module could not be found. '
      'Ensure you have downloaded it from '
      'https://github.com/google-research/google-research/tree/master/social_rl/gym_multigrid'
      ' before running this example.') from err

# Disables verbose np.bool warnings that occur in multigrid.
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    message='`np.bool` is a deprecated alias')


class MultigridWrapper(dm_env.Environment):
  """Environment wrapper for Multigrid environments.

  Note: the main difference with vanilla GymWrapper is that reward_spec() is
  overridden and rewards are cast to np.arrays in step()
  """

  def __init__(self, environment: multigrid.MultiGridEnv):
    """Initializes environment.

    Args:
      environment: the environment.
    """
    self._environment = environment
    self._reset_next_step = True
    self._last_info = None
    self.num_agents = environment.n_agents  # pytype: disable=attribute-error

    # Convert action and observation specs.
    obs_space = self._environment.observation_space
    act_space = self._environment.action_space
    self._observation_spec = _convert_to_spec(
        obs_space, self.num_agents, name='observation')
    self._action_spec = _convert_to_spec(
        act_space, self.num_agents, name='action')

  def process_obs(self, observation: types.NestedArray) -> types.NestedArray:
    # Convert observations to agent-index-first format
    observation = dict_obs_to_list_obs(observation)

    # Assign dtypes to multigrid observations (some of which are lists by
    # default, so do not have a precise dtype that matches their observation
    # spec. This ensures no replay signature mismatch issues occur).
    observation = tree.map_structure(lambda x, t: np.asarray(x, dtype=t.dtype),
                                     observation, self.observation_spec())
    return observation

  def reset(self) -> dm_env.TimeStep:
    """Resets the episode."""
    self._reset_next_step = False
    observation = self.process_obs(self._environment.reset())

    # Reset the diagnostic information.
    self._last_info = None
    return dm_env.restart(observation)

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    """Steps the environment."""
    if self._reset_next_step:
      return self.reset()

    observation, reward, done, info = self._environment.step(action)
    observation = self.process_obs(observation)

    self._reset_next_step = done
    self._last_info = info

    def _map_reward_spec(x, t):
      if np.isscalar(x):
        return t.dtype.type(x)
      return np.asarray(x, dtype=t.dtype)

    reward = tree.map_structure(
        _map_reward_spec,
        reward,
        self.reward_spec())

    if done:
      truncated = info.get('TimeLimit.truncated', False)
      if truncated:
        return dm_env.truncation(reward, observation)
      return dm_env.termination(reward, observation)
    return dm_env.transition(reward, observation)

  def observation_spec(self) -> types.NestedSpec:  # Internal pytype check.
    return self._observation_spec

  def action_spec(self) -> types.NestedSpec:  # Internal pytype check.
    return self._action_spec

  def reward_spec(self) -> types.NestedSpec:  # Internal pytype check.
    return [specs.Array(shape=(), dtype=float, name='rewards')
           ] * self._environment.n_agents

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

  def __getattr__(self, name: str) -> Any:
    """Returns any other attributes of the underlying environment."""
    return getattr(self._environment, name)

  def close(self):
    self._environment.close()


def _get_single_agent_spec(spec):
  """Returns a single-agent spec from multiagent multigrid spec.

  Primarily used for converting multigrid specs to multiagent Acme specs,
  wherein actions and observations specs are expected to be lists (each entry
  corresponding to the spec of that particular agent). Note that this function
  assumes homogeneous observation / action specs across all agents, which is the
  case in multigrid.

  Args:
    spec: multigrid environment spec.
  """
  def make_single_agent_spec(spec):
    if not spec.shape:  # Rewards & discounts
      shape = ()
    elif len(spec.shape) == 1:  # Actions
      shape = ()
    else:  # Observations
      shape = spec.shape[1:]

    if isinstance(spec, specs.BoundedArray):
      # Bounded rewards and discounts often have no dimensions as they are
      # amongst the agents, whereas observations are of shape [num_agents, ...].
      # The following pair of if statements handle both cases accordingly.
      minimum = spec.minimum if spec.minimum.ndim == 0 else spec.minimum[0]
      maximum = spec.maximum if spec.maximum.ndim == 0 else spec.maximum[0]
      return specs.BoundedArray(
          shape=shape,
          name=spec.name,
          minimum=minimum,
          maximum=maximum,
          dtype=spec.dtype)
    elif isinstance(spec, specs.DiscreteArray):
      return specs.DiscreteArray(
          num_values=spec.num_values, dtype=spec.dtype, name=spec.name)
    elif isinstance(spec, specs.Array):
      return specs.Array(shape=shape, dtype=spec.dtype, name=spec.name)
    else:
      raise ValueError(f'Unexpected spec type {type(spec)}.')

  single_agent_spec = jax.tree_map(make_single_agent_spec, spec)
  return single_agent_spec


def _gym_to_spec(space: gym.Space,
                 name: Optional[str] = None) -> types.NestedSpec:
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
        name=name)

  elif isinstance(space, spaces.MultiBinary):
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=0.0,
        maximum=1.0,
        name=name)

  elif isinstance(space, spaces.MultiDiscrete):
    return specs.BoundedArray(
        shape=space.shape,
        dtype=space.dtype,
        minimum=np.zeros(space.shape),
        maximum=space.nvec - 1,
        name=name)

  elif isinstance(space, spaces.Tuple):
    return tuple(_gym_to_spec(s, name) for s in space.spaces)

  elif isinstance(space, spaces.Dict):
    return {
        key: _gym_to_spec(value, key) for key, value in space.spaces.items()
    }

  else:
    raise ValueError('Unexpected gym space: {}'.format(space))


def _convert_to_spec(space: gym.Space,
                     num_agents: int,
                     name: Optional[str] = None) -> types.NestedSpec:
  """Converts multigrid Gym space to an Acme multiagent spec.

  Args:
    space: The Gym space to convert.
    num_agents: the number of agents.
    name: Optional name to apply to all return spec(s).

  Returns:
    A dm_env spec or nested structure of specs, corresponding to the input
    space.
  """
  # Convert gym specs to acme specs
  spec = _gym_to_spec(space, name)
  # Then change spec indexing from observation-key-first to agent-index-first
  return [_get_single_agent_spec(spec)] * num_agents


def dict_obs_to_list_obs(
    observation: types.NestedArray
) -> List[Dict[ma_types.AgentID, types.NestedArray]]:
  """Returns multigrid observations converted to agent-index-first format.

  By default, multigrid observations are structured as:
    observation['image'][agent_index]
    observation['direction'][agent_index]
    ...

  However, multiagent Acme expects observations with agent indices first:
    observation[agent_index]['image']
    observation[agent_index]['direction']

  This function simply converts multigrid observations to the latter format.

  Args:
    observation:
  """
  return [dict(zip(observation, v)) for v in zip(*observation.values())]


def make_multigrid_environment(
    env_name: str = 'MultiGrid-Empty-5x5-v0') -> dm_env.Environment:
  """Returns Multigrid Multiagent Gym environment.

  Args:
    env_name: name of multigrid task. See social_rl.gym_multigrid.envs for the
      available environments.
  """
  # Load the gym environment.
  env = gym.make(env_name)

  # Make sure the environment obeys the dm_env.Environment interface.
  env = MultigridWrapper(env)
  env = wrappers.SinglePrecisionWrapper(env)
  env = multiagent_dict_key_wrapper.MultiagentDictKeyWrapper(env)
  return env
