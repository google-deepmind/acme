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

"""An environment wrapper to produce pixel observations from dm_control."""

import collections
from acme.wrappers import base
from dm_control.rl import control
from dm_control.suite.wrappers import pixels  # type: ignore
import dm_env


class MujocoPixelWrapper(base.EnvironmentWrapper):
  """Produces pixel observations from Mujoco environment observations."""

  def __init__(self,
               environment: control.Environment,
               *,
               height: int = 84,
               width: int = 84,
               camera_id: int = 0):
    render_kwargs = {'height': height, 'width': width, 'camera_id': camera_id}
    pixel_environment = pixels.Wrapper(
        environment, pixels_only=True, render_kwargs=render_kwargs)
    super().__init__(pixel_environment)

  def step(self, action) -> dm_env.TimeStep:
    return self._convert_timestep(self._environment.step(action))

  def reset(self) -> dm_env.TimeStep:
    return self._convert_timestep(self._environment.reset())

  def observation_spec(self):
    return self._environment.observation_spec()['pixels']

  def _convert_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    """Removes the pixel observation's OrderedDict wrapper."""
    observation: collections.OrderedDict = timestep.observation
    return timestep._replace(observation=observation['pixels'])
