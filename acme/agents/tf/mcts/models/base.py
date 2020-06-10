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

"""Base model class, specifying the interface.."""

import abc

from acme.agents.tf.mcts import types

import dm_env


class Model(dm_env.Environment, abc.ABC):
  """Base (abstract) class for models used for planning via MCTS."""

  @abc.abstractmethod
  def load_checkpoint(self):
    """Loads a saved model state, if it exists."""

  @abc.abstractmethod
  def save_checkpoint(self):
    """Saves the model state so that we can reset it after a rollout."""

  @abc.abstractmethod
  def update(
      self,
      timestep: dm_env.TimeStep,
      action: types.Action,
      next_timestep: dm_env.TimeStep,
  ) -> dm_env.TimeStep:
    """Updates the model given an observation, action, reward, and discount."""

  @abc.abstractmethod
  def reset(self, initial_state: types.Observation = None):
    """Resets the model, optionally to an initial state."""

  @property
  @abc.abstractmethod
  def needs_reset(self) -> bool:
    """Returns whether or not the model needs to be reset."""
