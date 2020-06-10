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

"""A simulator model, which wraps a copy of the true environment."""

import copy

from acme.agents.tf.mcts import types
from acme.agents.tf.mcts.models import base

import dm_env


class Simulator(base.Model):
  """A simulator model, which wraps a copy of the true environment.

  Assumes that the environment (including RNG) is fully copyable via `deepcopy`.
  """

  def __init__(self, env: dm_env.Environment):
    # Make a 'checkpoint' copy env to save/load from when doing rollouts.
    self._env = copy.deepcopy(env)
    self._checkpoint = copy.deepcopy(env)
    self._needs_reset = True

  def update(
      self,
      timestep: dm_env.TimeStep,
      action: types.Action,
      next_timestep: dm_env.TimeStep,
  ) -> dm_env.TimeStep:
    # Call update() once per 'real' experience to keep this env in sync.
    return self.step(action)

  def save_checkpoint(self):
    self._checkpoint = copy.deepcopy(self._env)

  def load_checkpoint(self):
    self._env = copy.deepcopy(self._checkpoint)

  def step(self, action: types.Action) -> dm_env.TimeStep:
    if self._needs_reset:
      raise ValueError('This model needs to be explicitly reset.')
    return self._env.step(action)

  def reset(self, *unused_args, **unused_kwargs):
    self._needs_reset = False
    return self._env.reset()

  def observation_spec(self):
    return self._env.observation_spec()

  def action_spec(self):
    return self._env.action_spec()

  @property
  def needs_reset(self) -> bool:
    return self._needs_reset
