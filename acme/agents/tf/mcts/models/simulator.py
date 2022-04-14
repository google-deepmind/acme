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
import dataclasses

from acme.agents.tf.mcts import types
from acme.agents.tf.mcts.models import base
import dm_env


@dataclasses.dataclass
class Checkpoint:
  """Holds the checkpoint state for the environment simulator."""
  needs_reset: bool
  environment: dm_env.Environment


class Simulator(base.Model):
  """A simulator model, which wraps a copy of the true environment.

  Assumptions:
    - The environment (including RNG) is fully copyable via `deepcopy`.
    - Environment dynamics (modulo episode resets) are deterministic.
  """

  _checkpoint: Checkpoint
  _env: dm_env.Environment

  def __init__(self, env: dm_env.Environment):
    # Make a 'checkpoint' copy env to save/load from when doing rollouts.
    self._env = copy.deepcopy(env)
    self._needs_reset = True
    self.save_checkpoint()

  def update(
      self,
      timestep: dm_env.TimeStep,
      action: types.Action,
      next_timestep: dm_env.TimeStep,
  ) -> dm_env.TimeStep:
    # Call update() once per 'real' experience to keep this env in sync.
    return self.step(action)

  def save_checkpoint(self):
    self._checkpoint = Checkpoint(
        needs_reset=self._needs_reset,
        environment=copy.deepcopy(self._env),
    )

  def load_checkpoint(self):
    self._env = copy.deepcopy(self._checkpoint.environment)
    self._needs_reset = self._checkpoint.needs_reset

  def step(self, action: types.Action) -> dm_env.TimeStep:
    if self._needs_reset:
      raise ValueError('This model needs to be explicitly reset.')
    timestep = self._env.step(action)
    self._needs_reset = timestep.last()
    return timestep

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
