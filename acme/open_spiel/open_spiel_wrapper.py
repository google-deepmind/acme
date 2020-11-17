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

"""Wraps an OpenSpiel RL environment to be used as a dm_env environment."""

from typing import List

from acme import specs
from acme import types
import dm_env
import numpy as np
from open_spiel.python import rl_environment
import pyspiel


# TODO Wrap the underlying OpenSpiel game directly instead of OpenSpiel's
# rl_environment?
class OpenSpielWrapper(dm_env.Environment):
  """Environment wrapper for OpenSpiel RL environments."""

  # Note: we don't inherit from base.EnvironmentWrapper because that class
  # assumes that the wrapped environment is a dm_env.Environment.

  def __init__(self, environment: rl_environment.Environment):
    self._environment = environment
    self._reset_next_step = True
    assert environment._game.get_type(
    ).dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL, (
        "Currently only supports sequential games.")

  def reset(self) -> dm_env.TimeStep:
    """Resets the episode."""
    self._reset_next_step = False
    open_spiel_timestep = self._environment.reset()
    observation = self._convert_observation(open_spiel_timestep.observations)
    assert open_spiel_timestep.step_type == rl_environment.StepType.FIRST
    return dm_env.restart(observation)

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    """Steps the environment."""
    if self._reset_next_step:
      return self.reset()

    open_spiel_timestep = self._environment.step(action)

    if open_spiel_timestep.step_type == rl_environment.StepType.LAST:
      self._reset_next_step = True

    observation = self._convert_observation(open_spiel_timestep.observations)
    reward = np.asarray(open_spiel_timestep.rewards)
    discount = np.asarray(open_spiel_timestep.discounts)
    step_type = open_spiel_timestep.step_type

    if step_type == rl_environment.StepType.FIRST:
      step_type = dm_env.StepType.FIRST
    elif step_type == rl_environment.StepType.MID:
      step_type = dm_env.StepType.MID
    elif step_type == rl_environment.StepType.LAST:
      step_type = dm_env.StepType.LAST
    else:
      raise ValueError("Did not recognize OpenSpiel StepType")

    return dm_env.TimeStep(observation=observation,
                           reward=reward,
                           discount=discount,
                           step_type=step_type)

  # TODO Convert OpenSpiel observation so it's dm_env compatible. Dm_env
  # timesteps allow for dicts and nesting, but they require the leaf elements
  # be numpy arrays, whereas OpenSpiel timestep leaf elements are python lists.
  # Also, the list of legal actions must be converted to a legal actions mask.
  def _convert_observation(
      self, open_spiel_observation: types.NestedArray) -> types.NestedArray:
    observation = {"info_state": [], "legal_actions": [], "current_player": []}
    info_state = []
    for player_info_state in open_spiel_observation["info_state"]:
      info_state.append(np.asarray(player_info_state))
    observation["info_state"] = info_state
    legal_actions = []
    for indicies in open_spiel_observation["legal_actions"]:
      legals = np.zeros(self._environment._game.num_distinct_actions())
      legals[indicies] = 1
      legal_actions.append(legals)
    observation["legal_actions"] = legal_actions
    observation["current_player"] = self._environment._state.current_player()
    return observation

  # TODO These specs describe the timestep that the actor and learner ultimately
  # receive, not the timestep that gets passed to the OpenSpiel agent. See
  # acme/open_spiel/agents/agent.py for more details.
  def observation_spec(self) -> types.NestedSpec:
    if self._environment._use_observation:
      return specs.Array((self._environment._game.observation_tensor_size(),),
                         np.float32)
    else:
      return specs.Array(
          (self._environment._game.information_state_tensor_size(),),
          np.float32)

  def action_spec(self) -> types.NestedSpec:
    return specs.DiscreteArray(self._environment._game.num_distinct_actions())

  def reward_spec(self) -> types.NestedSpec:
    return specs.BoundedArray((),
                              np.float32,
                              minimum=self._game.min_utility(),
                              maximum=self._game.max_utility())

  def discount_spec(self) -> types.NestedSpec:
    return specs.BoundedArray((), np.float32, minimum=0, maximum=1.0)

  def legal_actions_spec(self) -> types.NestedSpec:
    return specs.BoundedArray((self._environment._game.num_distinct_actions(),),
                              np.float32,
                              minimum=0,
                              maximum=1.0)

  def terminals_spec(self) -> types.NestedSpec:
    return specs.BoundedArray((), np.float32, minimum=0, maximum=1.0)

  @property
  def environment(self):
    """Returns the wrapped environment."""
    return self._environment

  def __getattr__(self, name: str):
    # Expose any other attributes of the underlying environment.
    return getattr(self._environment, name)
