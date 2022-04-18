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

import os.path
from typing import Callable, Optional, Sequence, Tuple, Union
from acme.utils import paths
from acme.wrappers import base
import dm_env

import matplotlib
matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
import matplotlib.animation as anim  # pylint: disable=g-import-not-at-top
import matplotlib.pyplot as plt
import numpy as np

# Internal imports.
# Make sure you have FFMpeg configured.

def make_animation(frames: Sequence[np.ndarray], frame_rate: float,
                   figsize: Union[float, Tuple[int, int]]) -> anim.Animation:
  """Generates an animation from a stack of frames."""

  # Set animation characteristics.
  if figsize is None:
    height, width, _ = frames[0].shape
  elif isinstance(figsize, tuple):
    height, width = figsize
  else:
    diagonal = figsize
    height, width, _ = frames[0].shape
    scale_factor = diagonal / np.sqrt(height**2 + width**2)
    width *= scale_factor
    height *= scale_factor

  dpi = 70
  interval = int(round(1e3 / frame_rate))  # Time (in ms) between frames.

  # Create and configure the figure.
  fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
  ax.set_axis_off()
  ax.set_aspect('equal')
  ax.set_position([0, 0, 1, 1])

  # Initialize the first frame.
  im = ax.imshow(frames[0])

  # Create the function that will modify the frame, creating an animation.
  def update(frame):
    im.set_data(frame)
    return [im]

  return anim.FuncAnimation(
      fig=fig,
      func=update,
      frames=frames,
      interval=interval,
      blit=True,
      repeat=False)


class VideoWrapper(base.EnvironmentWrapper):
  """Wrapper which creates and records videos from generated observations.

  This will limit itself to recording once every `record_every` episodes and
  videos will be recorded to the directory `path` + '/<unique id>/videos' where
  `path` defaults to '~/acme'. Users can specify the size of the screen by
  passing either a tuple giving height and width or a float giving the size
  of the diagonal.
  """

  def __init__(self,
               environment: dm_env.Environment,
               *,
               path: str = '~/acme',
               filename: str = '',
               process_path: Callable[[str, str], str] = paths.process_path,
               record_every: int = 100,
               frame_rate: int = 30,
               figsize: Optional[Union[float, Tuple[int, int]]] = None):
    super(VideoWrapper, self).__init__(environment)
    self._path = process_path(path, 'videos')
    self._filename = filename
    self._record_every = record_every
    self._frame_rate = frame_rate
    self._frames = []
    self._counter = 0
    self._figsize = figsize

  def _render_frame(self, observation):
    """Renders a frame from the given environment observation."""
    return observation

  def _write_frames(self):
    """Writes frames to video."""
    if self._counter % self._record_every == 0:
      path = os.path.join(self._path,
                          f'{self._filename}_{self._counter:04d}.html')
      video = make_animation(self._frames, self._frame_rate,
                             self._figsize).to_html5_video()

      with open(path, 'w') as f:
        f.write(video)

    # Clear the frame buffer whether a video was generated or not.
    self._frames = []

  def _append_frame(self, observation):
    """Appends a frame to the sequence of frames."""
    if self._counter % self._record_every == 0:
      self._frames.append(self._render_frame(observation))

  def step(self, action) -> dm_env.TimeStep:
    timestep = self.environment.step(action)
    self._append_frame(timestep.observation)
    return timestep

  def reset(self) -> dm_env.TimeStep:
    # If the frame buffer is nonempty, flush it and record video
    if self._frames:
      self._write_frames()
    self._counter += 1
    timestep = self.environment.reset()
    self._append_frame(timestep.observation)
    return timestep

  def make_html_animation(self):
    if self._frames:
      return make_animation(self._frames, self._frame_rate,
                            self._figsize).to_html5_video()
    else:
      raise ValueError('make_html_animation should be called after running a '
                       'trajectory and before calling reset().')

  def close(self):
    if self._frames:
      self._write_frames()
      self._frames = []
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

  def _render_frame(self, unused_observation):
    del unused_observation

    # We've checked above that this attribute should exist. Pytype won't like
    # it if we just try and do self.environment.physics, so we use the slightly
    # grosser version below.
    physics = getattr(self.environment, 'physics')

    if self._camera_id is not None:
      frame = physics.render(
          camera_id=self._camera_id, height=self._height, width=self._width)
    else:
      # If camera_id is None, we create a minimal canvas that will accommodate
      # physics.model.ncam frames, and render all of them on a grid.
      num_cameras = physics.model.ncam
      num_columns = int(np.ceil(np.sqrt(num_cameras)))
      num_rows = int(np.ceil(float(num_cameras)/num_columns))
      height = self._height
      width = self._width

      # Make a black canvas.
      frame = np.zeros((num_rows*height, num_columns*width, 3), dtype=np.uint8)

      for col in range(num_columns):
        for row in range(num_rows):

          camera_id = row*num_columns + col

          if camera_id >= num_cameras:
            break

          subframe = physics.render(
              camera_id=camera_id, height=height, width=width)

          # Place the frame in the appropriate rectangle on the pixel canvas.
          frame[row*height:(row+1)*height, col*width:(col+1)*width] = subframe

    return frame
