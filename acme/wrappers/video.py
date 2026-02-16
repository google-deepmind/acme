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

"""Environment wrappers which record videos.

The code used to generate animations in this wrapper is based on that used in
the `dm_control/tutorial.ipynb` file.
"""

from typing import Callable, Optional, Sequence, Tuple, Union

from absl import logging
from acme.utils import paths
from acme.wrappers import base
import dm_env
import numpy as np


def make_animation(
    frames: Sequence[np.ndarray],
    frame_rate: float,
    figsize: Optional[Union[float, Tuple[int, int]]],
):
  """Generates a matplotlib animation from a stack of frames."""
  logging.warning(
      'make_animation is deprecated and currently acts as a no-op in order to '
      'avoid using ffmpeg directly. The old behavior can be restored by '
      'replacing the direct call to ffmpeg within matplotlib.'
  )
  del frames
  del frame_rate
  del figsize
  return None


class VideoWrapper(base.EnvironmentWrapper):
  """Wrapper which creates and records videos from generated observations.

  This will limit itself to recording once every `record_every` episodes and
  videos will be recorded to the directory `path` + '/<unique id>/videos' where
  `path` defaults to '~/acme'. Users can specify the size of the screen by
  passing either a tuple giving height and width or a float giving the size
  of the diagonal.
  """

  def __init__(
      self,
      environment: dm_env.Environment,
      *,
      path: str = '~/acme',
      filename: str = '',
      process_path: Callable[[str, str], str] = paths.process_path,
      record_every: int = 100,
      frame_rate: int = 30,
      figsize: Optional[Union[float, Tuple[int, int]]] = None,
      to_html: bool = True,
  ):
    logging.warning(
        'VideoWrapper is deprecated and currently acts as a no-op in order to '
        'avoid using ffmpeg directly. The old behavior can be restored by '
        'replacing the direct call to ffmpeg within matplotlib.'
    )
    super(VideoWrapper, self).__init__(environment)

  def step(self, action) -> dm_env.TimeStep:
    return self.environment.step(action)

  def reset(self) -> dm_env.TimeStep:
    return self.environment.reset()

  def make_html_animation(self):
    return None

  def close(self):
    self.environment.close()


class MujocoVideoWrapper(VideoWrapper):
  """VideoWrapper which generates videos from a mujoco physics object.

  This passes its keyword arguments into the parent `VideoWrapper` class (refer
  here for any default arguments).
  """

  # Note that since we can be given a wrapped mujoco environment we can't give
  # the type as dm_control.Environment.

  def __init__(self,
               environment: dm_env.Environment,
               *,
               frame_rate: Optional[int] = None,
               camera_id: Optional[int] = 0,
               height: int = 240,
               width: int = 320,
               playback_speed: float = 1.,
               **kwargs):

    # Check that we have a mujoco environment (or a wrapper thereof).
    if not hasattr(environment, 'physics'):
      raise ValueError('MujocoVideoWrapper expects an environment which '
                       'exposes a physics attribute corresponding to a MuJoCo '
                       'physics engine')

    # Compute frame rate if not set.
    if frame_rate is None:
      try:
        control_timestep = getattr(environment, 'control_timestep')()
      except AttributeError as e:
        raise AttributeError('MujocoVideoWrapper expects an environment which '
                             'exposes a control_timestep method, like '
                             'dm_control environments, or frame_rate '
                             'to be specified.') from e
      frame_rate = int(round(playback_speed / control_timestep))

    super().__init__(environment, frame_rate=frame_rate, **kwargs)
    self._camera_id = camera_id
    self._height = height
    self._width = width
