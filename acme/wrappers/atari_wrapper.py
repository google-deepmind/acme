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

"""Atari wrappers functionality for Python environments."""

import abc
from typing import Tuple, List, Optional, Sequence, Union

from acme.wrappers import base
from acme.wrappers import frame_stacking

import dm_env
from dm_env import specs
import numpy as np
from PIL import Image

RGB_INDEX = 0  # Observation index holding the RGB data.
LIVES_INDEX = 1  # Observation index holding the lives count.
NUM_COLOR_CHANNELS = 3  # Number of color channels in RGB data.


class BaseAtariWrapper(abc.ABC, base.EnvironmentWrapper):
  """Abstract base class for Atari wrappers.

  This assumes that the input environment is a dm_env.Environment instance in
  which observations are tuples whose first element is an RGB observation and
  the second element is the lives count.

  The wrapper itself performs the following modifications:

    1. Soft-termination (setting discount to zero) on loss of life.
    2. Action repeats.
    3. Frame pooling for action repeats.
    4. Conversion to grayscale and downscaling.
    5. Reward clipping.
    6. Observation stacking.

  The details of grayscale conversion, downscaling, and frame pooling are
  delegated to the concrete subclasses.

  This wrapper will raise an error if the underlying Atari environment does not:

  - Exposes RGB observations in interleaved format (shape `(H, W, C)`).
  - Expose zero-indexed actions.

  Note that this class does not expose a configurable rescale method (defaults
  to bilinear internally).

  This class also exposes an additional option `to_float` that doesn't feature
  in other wrappers, which rescales pixel values to floats in the range [0, 1].
  """

  def __init__(self,
               environment: dm_env.Environment,
               *,
               max_abs_reward: Optional[float] = None,
               scale_dims: Optional[Tuple[int, int]] = (84, 84),
               action_repeats: int = 4,
               pooled_frames: int = 2,
               zero_discount_on_life_loss: bool = False,
               expose_lives_observation: bool = False,
               num_stacked_frames: int = 4,
               max_episode_len: Optional[int] = None,
               to_float: bool = False,
               grayscaling: bool = True):
    """Initializes a new AtariWrapper.

    Args:
      environment: An Atari environment.
      max_abs_reward: Maximum absolute reward value before clipping is applied.
        If set to `None` (default), no clipping is applied.
      scale_dims: Image size for the rescaling step after grayscaling, given as
        `(height, width)`. Set to `None` to disable resizing.
      action_repeats: Number of times to step wrapped environment for each given
        action.
      pooled_frames: Number of observations to pool over. Set to 1 to disable
        frame pooling.
      zero_discount_on_life_loss: If `True`, sets the discount to zero when the
        number of lives decreases in in Atari environment.
      expose_lives_observation: If `False`, the `lives` part of the observation
        is discarded, otherwise it is kept as part of an observation tuple. This
        does not affect the `zero_discount_on_life_loss` feature. When enabled,
        the observation consists of a single pixel array, otherwise it is a
        tuple (pixel_array, lives).
      num_stacked_frames: Number of recent (pooled) observations to stack into
        the returned observation.
      max_episode_len: Number of frames before truncating episode. By default,
        there is no maximum length.
      to_float: If `True`, rescales RGB observations to floats in [0, 1].
      grayscaling: If `True` returns a grayscale version of the observations. In
        this case, the observation is 3D (H, W, num_stacked_frames). If `False`
        the observations are RGB and have shape (H, W, C, num_stacked_frames).

    Raises:
      ValueError: For various invalid inputs.
    """
    if not 1 <= pooled_frames <= action_repeats:
      raise ValueError("pooled_frames ({}) must be between 1 and "
                       "action_repeats ({}) inclusive".format(
                           pooled_frames, action_repeats))

    if zero_discount_on_life_loss:
      super().__init__(_ZeroDiscountOnLifeLoss(environment))
    else:
      super().__init__(environment)

    if not max_episode_len:
      max_episode_len = np.inf

    self._frame_stacker = frame_stacking.FrameStacker(
        num_frames=num_stacked_frames)
    self._action_repeats = action_repeats
    self._pooled_frames = pooled_frames
    self._scale_dims = scale_dims
    self._max_abs_reward = max_abs_reward or np.inf
    self._to_float = to_float
    self._expose_lives_observation = expose_lives_observation

    if scale_dims:
      self._height, self._width = scale_dims
    else:
      spec = environment.observation_spec()
      self._height, self._width = spec[RGB_INDEX].shape[:2]

    self._episode_len = 0
    self._max_episode_len = max_episode_len
    self._reset_next_step = True

    self._grayscaling = grayscaling

    # Based on underlying observation spec, decide whether lives are to be
    # included in output observations.
    observation_spec = self._environment.observation_spec()
    spec_names = [spec.name for spec in observation_spec]
    if "lives" in spec_names and spec_names.index("lives") != 1:
      raise ValueError("`lives` observation needs to have index 1 in Atari.")

    self._observation_spec = self._init_observation_spec()

    self._raw_observation = None

  def _init_observation_spec(self):
    """Computes the observation spec for the pixel observations.

    Returns:
      An `Array` specification for the pixel observations.
    """
    if self._to_float:
      pixels_dtype = float
    else:
      pixels_dtype = np.uint8

    if self._grayscaling:
      pixels_spec_shape = (self._height, self._width)
      pixels_spec_name = "grayscale"
    else:
      pixels_spec_shape = (self._height, self._width, NUM_COLOR_CHANNELS)
      pixels_spec_name = "RGB"

    pixel_spec = specs.Array(
        shape=pixels_spec_shape, dtype=pixels_dtype, name=pixels_spec_name)
    pixel_spec = self._frame_stacker.update_spec(pixel_spec)

    if self._expose_lives_observation:
      return (pixel_spec,) + self._environment.observation_spec()[1:]
    return pixel_spec

  def reset(self) -> dm_env.TimeStep:
    """Resets environment and provides the first timestep."""
    self._reset_next_step = False
    self._episode_len = 0
    self._frame_stacker.reset()
    timestep = self._environment.reset()

    observation = self._observation_from_timestep_stack([timestep])

    return self._postprocess_observation(
        timestep._replace(observation=observation))

  def step(self, action: int) -> dm_env.TimeStep:
    """Steps up to action_repeat times and returns a post-processed step."""
    if self._reset_next_step:
      return self.reset()

    timestep_stack = []

    # Step on environment multiple times for each selected action.
    for _ in range(self._action_repeats):
      timestep = self._environment.step([np.array([action])])

      self._episode_len += 1
      if self._episode_len == self._max_episode_len:
        timestep = timestep._replace(step_type=dm_env.StepType.LAST)

      timestep_stack.append(timestep)

      if timestep.last():
        # Action repeat frames should not span episode boundaries. Also, no need
        # to pad with zero-valued observations as all the reductions in
        # _postprocess_observation work gracefully for any non-zero size of
        # timestep_stack.
        self._reset_next_step = True
        break

    # Determine a single step type. We let FIRST take priority over LAST, since
    # we think it's more likely algorithm code will be set up to deal with that,
    # due to environments supporting reset() (which emits a FIRST).
    # Note we'll never have LAST then FIRST in timestep_stack here.
    step_type = dm_env.StepType.MID
    for timestep in timestep_stack:
      if timestep.first():
        step_type = dm_env.StepType.FIRST
        break
      elif timestep.last():
        step_type = dm_env.StepType.LAST
        break

    if timestep_stack[0].first():
      # Update first timestep to have identity effect on reward and discount.
      timestep_stack[0] = timestep_stack[0]._replace(reward=0., discount=1.)

    # Sum reward over stack.
    reward = sum(timestep_t.reward for timestep_t in timestep_stack)

    # Multiply discount over stack (will either be 0. or 1.).
    discount = np.product(
        [timestep_t.discount for timestep_t in timestep_stack])

    observation = self._observation_from_timestep_stack(timestep_stack)

    timestep = dm_env.TimeStep(
        step_type=step_type,
        reward=reward,
        observation=observation,
        discount=discount)

    return self._postprocess_observation(timestep)

  @abc.abstractmethod
  def _preprocess_pixels(self, timestep_stack: List[dm_env.TimeStep]):
    """Process Atari pixels."""

  def _observation_from_timestep_stack(self,
                                       timestep_stack: List[dm_env.TimeStep]):
    """Compute the observation for a stack of timesteps."""
    self._raw_observation = timestep_stack[-1].observation[RGB_INDEX].copy()
    processed_pixels = self._preprocess_pixels(timestep_stack)

    if self._to_float:
      stacked_observation = self._frame_stacker.step(processed_pixels / 255.0)
    else:
      stacked_observation = self._frame_stacker.step(processed_pixels)

    # We use last timestep for lives only.
    observation = timestep_stack[-1].observation
    if self._expose_lives_observation:
      return (stacked_observation,) + observation[1:]

    return stacked_observation

  def _postprocess_observation(self,
                               timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    """Observation processing applied after action repeat consolidation."""

    if timestep.first():
      return dm_env.restart(timestep.observation)

    reward = np.clip(timestep.reward, -self._max_abs_reward,
                     self._max_abs_reward)

    return timestep._replace(reward=reward)

  def action_spec(self) -> specs.DiscreteArray:
    raw_spec = self._environment.action_spec()[0]
    return specs.DiscreteArray(num_values=raw_spec.maximum.item() -
                               raw_spec.minimum.item() + 1)

  def observation_spec(self) -> Union[specs.Array, Sequence[specs.Array]]:
    return self._observation_spec

  def reward_spec(self) -> specs.Array:
    return specs.Array(shape=(), dtype=float)

  @property
  def raw_observation(self) -> np.ndarray:
    """Returns the raw observation, after any pooling has been applied."""
    return self._raw_observation


class AtariWrapper(BaseAtariWrapper):
  """Standard "Nature Atari" wrapper for Python environments.

  Before being fed to a neural network, Atari frames go through a prepocessing,
  implemented in this wrapper. For historical reasons, there were different
  choices in the method to apply there between what was done in the Dopamine
  library and what is done in Acme. During the processing of
  Atari frames, three operations need to happen. Images are
  transformed from RGB to grayscale, we perform a max-pooling on the time scale,
  and images are resized to 84x84.

  1. The `standard` style (this one, matches previous acme versions):
     - does max pooling, then rgb -> grayscale
     - uses Pillow inter area interpolation for resizing
  2. The `dopamine` style:
     - does rgb -> grayscale, then max pooling
     - uses opencv bilinear interpolation for resizing.

  This can change the behavior of RL agents on some games. The recommended
  setting is to use the standard style with this class. The Dopamine setting is
  available in `atari_wrapper_dopamine.py` for the
  user that wishes to compare agents between librairies.
  """

  def _preprocess_pixels(self, timestep_stack: List[dm_env.TimeStep]):
    """Preprocess Atari frames."""
    # 1. Max pooling
    processed_pixels = np.max(
        np.stack([
            s.observation[RGB_INDEX]
            for s in timestep_stack[-self._pooled_frames:]
        ]),
        axis=0)

    # 2. RGB to grayscale
    if self._grayscaling:
      processed_pixels = np.tensordot(processed_pixels,
                                      [0.299, 0.587, 1 - (0.299 + 0.587)],
                                      (-1, 0))

    # 3. Resize
    processed_pixels = processed_pixels.astype(np.uint8, copy=False)
    if self._scale_dims != processed_pixels.shape[:2]:
      processed_pixels = Image.fromarray(processed_pixels).resize(
          (self._width, self._height), Image.BILINEAR)
      processed_pixels = np.array(processed_pixels, dtype=np.uint8)

    return processed_pixels


class _ZeroDiscountOnLifeLoss(base.EnvironmentWrapper):
  """Implements soft-termination (zero discount) on life loss."""

  def __init__(self, environment: dm_env.Environment):
    """Initializes a new `_ZeroDiscountOnLifeLoss` wrapper.

    Args:
      environment: An Atari environment.

    Raises:
      ValueError: If the environment does not expose a lives observation.
    """
    super().__init__(environment)
    self._reset_next_step = True
    self._last_num_lives = None

  def reset(self) -> dm_env.TimeStep:
    timestep = self._environment.reset()
    self._reset_next_step = False
    self._last_num_lives = timestep.observation[LIVES_INDEX]
    return timestep

  def step(self, action: int) -> dm_env.TimeStep:
    if self._reset_next_step:
      return self.reset()

    timestep = self._environment.step(action)
    lives = timestep.observation[LIVES_INDEX]

    is_life_loss = True
    # We have a life loss when:
    # The wrapped environment is in a regular (MID) transition.
    is_life_loss &= timestep.mid()
    # Lives have decreased since last time `step` was called.
    is_life_loss &= lives < self._last_num_lives

    self._last_num_lives = lives
    if is_life_loss:
      return timestep._replace(discount=0.0)
    return timestep
