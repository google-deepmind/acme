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

"""OpenSpiel agent interface."""

from typing import List, Tuple

from acme import core
from acme import types
from acme import specs
from acme.agents import agent
from acme.open_spiel import open_spiel_specs
from acme.tf import utils as tf2_utils
# Internal imports.

import dm_env
import numpy as np
from open_spiel.python import rl_environment
import pyspiel
import tensorflow as tf


class OpenSpielAgent(agent.Agent):
  """Agent class which combines acting and learning."""

  def __init__(self,
               actor: core.Actor,
               learner: core.Learner,
               min_observations: int,
               observations_per_step: float,
               player_id: int,
               should_update: bool = True):
    self._player_id = player_id
    self._should_update = should_update
    self._observed_first = False
    self._prev_action = None
    super().__init__(actor=actor,
                     learner=learner,
                     min_observations=min_observations,
                     observations_per_step=observations_per_step)

  def set_update(self, should_update: bool):
    self._should_update = should_update

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    current_player = observation["current_player"]
    assert current_player == self._player_id
    player_observation = observation["info_state"][current_player]
    legal_actions = observation["legal_actions"][current_player]
    self._prev_action = self._actor.select_action(player_observation,
                                                  legal_actions)
    return self._prev_action

  # TODO Eventually remove? Currently used for debugging.
  def print_policy(self, observation: types.NestedArray) -> types.NestedArray:
    current_player = observation["current_player"]
    assert current_player == self._player_id
    player_observation = observation["info_state"][current_player]
    legal_actions = observation["legal_actions"][current_player]

    batched_observation = tf2_utils.add_batch_dim(player_observation)
    policy = self._actor._policy_network(batched_observation, legal_actions)
    tf.print("Policy: ", policy.probs, summarize=-1)

  def observe_first(self, timestep: dm_env.TimeStep):
    current_player = timestep.observation["current_player"]
    assert current_player == self._player_id
    timestep = dm_env.TimeStep(
        observation=timestep.observation["info_state"][current_player],
        reward=None,
        discount=None,
        step_type=dm_env.StepType.FIRST)
    self._actor.observe_first(timestep)
    self._observed_first = True

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    current_player = next_timestep.observation["current_player"]
    if current_player == self._player_id:
      if not self._observed_first:
        self.observe_first(next_timestep)
      else:
        next_timestep, extras = self._convert_timestep(next_timestep)
        self._num_observations += 1
        self._actor.observe(self._prev_action, next_timestep, extras)
        if self._should_update:
          super().update()
    # TODO Note: we must account for situations where the first obs is a
    # terminal state, e.g. if an opponent folds in poker before we get to act.
    elif current_player == pyspiel.PlayerId.TERMINAL and self._observed_first:
      next_timestep, extras = self._convert_timestep(next_timestep)
      self._num_observations += 1
      self._actor.observe(self._prev_action, next_timestep, extras)
      self._observed_first = False
      self._prev_action = None
      if self._should_update:
        super().update()
    else:
      # TODO We ignore observations not relevant to this agent.
      pass

  # TODO In order to avoid bookkeeping in the environment loop, OpenSpiel agents
  # receive full timesteps that contain information for all agents. Here we
  # extract the information specific to this agent.
  def _convert_timestep(
      self, timestep: dm_env.TimeStep
  ) -> Tuple[dm_env.TimeStep, open_spiel_specs.Extras]:
    legal_actions = timestep.observation["legal_actions"][self._player_id]
    terminal = np.array(timestep.last(), dtype=np.float32)
    extras = open_spiel_specs.Extras(legal_actions=legal_actions,
                                     terminals=terminal)
    converted_timestep = dm_env.TimeStep(
        observation=timestep.observation["info_state"][self._player_id],
        reward=timestep.reward[self._player_id],
        discount=timestep.discount[self._player_id],
        step_type=timestep.step_type)
    return converted_timestep, extras
