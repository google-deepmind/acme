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

from typing import NamedTuple

# pytype: disable=import-error
from acme import specs
from acme import types
import dm_env
import numpy as np
from open_spiel.python import rl_environment
import pyspiel


class OLT(NamedTuple):
  """Container for (observation, legal_actions, terminal) tuples."""
  observation: types.Nest
  legal_actions: types.Nest
  terminal: types.Nest


# TODO Wrap the underlying OpenSpiel game directly instead of OpenSpiel's
# rl_environment?
class OpenSpielWrapper(dm_env.Environment):
  """Environment wrapper for OpenSpiel RL environments."""

  # Note: we don't inherit from base.EnvironmentWrapper because that class
  # assumes that the wrapped environment is a dm_env.Environment.

  def __init__(self, environment: rl_environment.Environment):
    self._environment = environment
    self._reset_next_step = True
    if environment._game.get_type(
    ).dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
      raise ValueError("Currently only supports sequential games.")

  def reset(self) -> dm_env.TimeStep:
    """Resets the episode."""
    self._reset_next_step = False
    open_spiel_timestep = self._environment.reset()
    observations = self._convert_observation(open_spiel_timestep)
    return dm_env.restart(observations)

  def step(self, action: types.NestedArray) -> dm_env.TimeStep:
    """Steps the environment."""
    if self._reset_next_step:
      return self.reset()

    open_spiel_timestep = self._environment.step(action)

    if open_spiel_timestep.step_type == rl_environment.StepType.LAST:
      self._reset_next_step = True

    observations = self._convert_observation(open_spiel_timestep)
    rewards = np.asarray(open_spiel_timestep.rewards)
    discounts = np.asarray(open_spiel_timestep.discounts)
    step_type = open_spiel_timestep.step_type

    if step_type == rl_environment.StepType.FIRST:
      step_type = dm_env.StepType.FIRST
    elif step_type == rl_environment.StepType.MID:
      step_type = dm_env.StepType.MID
    elif step_type == rl_environment.StepType.LAST:
      step_type = dm_env.StepType.LAST
    else:
      raise ValueError(
          "Did not recognize OpenSpiel StepType: {}".format(step_type))

    return dm_env.TimeStep(observation=observations,
                           reward=rewards,
                           discount=discounts,
                           step_type=step_type)

  # Convert OpenSpiel observation so it's dm_env compatible. Also, the list
  # of legal actions must be converted to a legal actions mask.
  def _convert_observation(
      self, open_spiel_timestep: rl_environment.TimeStep) -> types.NestedArray:
    observations = []
    for pid in range(self._environment.num_players):
      legals = np.zeros(self._environment._game.num_distinct_actions())
      legals[open_spiel_timestep.observations["legal_actions"][pid]] = 1
      player_observation = OLT(observation=np.asarray(
          open_spiel_timestep.observations["info_state"][pid]),
                               legal_actions=legals,
                               terminal=np.asarray(
                                   [float(open_spiel_timestep.last())]))
      observations.append(player_observation)
    return observations

  def observation_spec(self) -> types.NestedSpec:
    # Observation spec depends on whether the OpenSpiel environment is using
    # observation/information_state tensors
    if self._environment._use_observation:
      return OLT(observation=specs.Array(
          (self._environment._game.observation_tensor_size(),), np.float32),
                 legal_actions=specs.Array(
                     (self._environment._game.num_distinct_actions(),),
                     np.float32),
                 terminal=specs.Array((1,), np.float32))
    else:
      return OLT(observation=specs.Array(
          (self._environment._game.information_state_tensor_size(),),
          np.float32),
                 legal_actions=specs.Array(
                     (self._environment._game.num_distinct_actions(),),
                     np.float32),
                 terminal=specs.Array((1,), np.float32))

  def action_spec(self) -> types.NestedSpec:
    return specs.DiscreteArray(self._environment._game.num_distinct_actions())

  def reward_spec(self) -> types.NestedSpec:
    return specs.BoundedArray((),
                              np.float32,
                              minimum=self._game.min_utility(),
                              maximum=self._game.max_utility())

  def discount_spec(self) -> types.NestedSpec:
    return specs.BoundedArray((), np.float32, minimum=0, maximum=1.0)

  @property
  def environment(self) -> rl_environment.Environment:
    """Returns the wrapped environment."""
    return self._environment

  @property
  def current_player(self) -> int:
    return self._environment._state.current_player()

  def __getattr__(self, name: str):
    """Expose any other attributes of the underlying environment."""
    return getattr(self._environment, name)