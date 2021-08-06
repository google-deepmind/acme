"""Standard "Nature Atari" wrapper functionality for Python environments."""
from typing import Tuple, List, Optional, Sequence, Union

from acme.wrappers import base
from acme.wrappers import frame_stacking
import dm_env
from dm_env import specs
import numpy as np
from PIL import Image

"""Standard "Nature Atari" wrapper functionality for Python environments."""
from typing import Tuple, List, Optional, Sequence, Union

from acme.wrappers import base
from acme.wrappers import frame_stacking
import dm_env
from dm_env import specs
import numpy as np
from PIL import Image

class AtariRAMWrapper(base.EnvironmentWrapper):
  """Standard "Nature Atari" wrapper for Python environments.
  This assumes that the input environment is a dm_env.Environment instance in
  which observations are tuples whose first element is an RGB observation and
  the second element is the lives count.
  The wrapper itself performs the following modifications:
    1. Action repeats.
    2. Frame pooling for action repeats.
    3. Reward clipping.
    4. Observation stacking.
  This wrapper will raise an error if the underlying Atari environment does not:
  - Exposes 128 byte RAM states as observations .
  - Expose zero-indexed actions.
  Note that this class does not expose a configurable rescale method (defaults
  to bilinear internally).
  """

  def __init__(self,
               environment: dm_env.Environment,
               *,
               max_abs_reward: Optional[float] = None,
               action_repeats: int = 1, # 4
               pooled_frames: int = 1, # 2
               num_stacked_frames: int = 1, # 4
               max_episode_len: Optional[int] = None):
    """Initializes a new AtariWrapper.
    Args:
      environment: An Atari environment.
      max_abs_reward: Maximum absolute reward value before clipping is applied.
        If set to `None` (default), no clipping is applied.
      action_repeats: Number of times to step wrapped environment for each given
        action.
      pooled_frames: Number of observations to pool over. Set to 1 to disable
        frame pooling.
      num_stacked_frames: Number of recent (pooled) observations to stack into
        the returned observation.
      max_episode_len: Number of frames before truncating episode. By default,
        there is no maximum length.
    Raises:
      ValueError: For various invalid inputs.
    """
    if not 1 <= pooled_frames <= action_repeats:
      raise ValueError("pooled_frames ({}) must be between 1 and "
                       "action_repeats ({}) inclusive".format(
                           pooled_frames, action_repeats))

    super().__init__(environment)

    if not max_episode_len:
      max_episode_len = np.inf

    self._frame_stacker = frame_stacking.FrameStacker(
        num_frames=num_stacked_frames)
    self._action_repeats = action_repeats
    self._pooled_frames = pooled_frames
    self._max_abs_reward = max_abs_reward or np.inf
    self._num_stacked_frames = num_stacked_frames

    spec = environment.observation_spec()

    self._episode_len = 0
    self._max_episode_len = max_episode_len
    self._reset_next_step = True

    # Based on underlying observation spec, decide whether lives are to be
    # included in output observations.
    observation_spec = self._environment.observation_spec()
    spec_names = [spec.name for spec in observation_spec]

    self._observation_spec = self._init_observation_spec()

    self._raw_observation = None

  def _init_observation_spec(self):
    """Computes the observation spec for the ram observations.
    Returns:
      An `Array` specification for the ram observations.
    """
    ram_dtype = np.uint8

    if self._num_stacked_frames == 1:
      ram_spec_shape = (128,)
    else:
      ram_spec_shape = (self._num_stacked_frames,128)
    
    ram_spec_name = "RAM"

    ram_spec = specs.Array(
        shape=ram_spec_shape, dtype=ram_dtype, name=ram_spec_name)
    # ram_spec = self._frame_stacker.update_spec(ram_spec)
    # print(ram_spec)
    # print(ram_spec.shape)
    return ram_spec

  def reset(self) -> dm_env.TimeStep:
    """Resets environment and provides the first timestep."""
    self._reset_next_step = False
    self._episode_len = 0
    self._frame_stacker.reset()
    timestep_stack = []
    timestep = self._environment.reset()
    timestep_stack.append(timestep)
    # print("timestep : ",timestep)
    for _ in range(self._action_repeats - 1):
      timestep = self._environment.step([np.array([0])])
      #print("timestep: {}".format(timestep.observation.shape))

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

  def _observation_on_reset(self, timestep: dm_env.TimeStep):
    """Computes the current observation after a reset.
    Args:
      timestep: `TimeStep` returned by the raw_environment during a reset.
    Returns:
      A stack of processed pixel frames.
    """
    observation = timestep.observation
    return observation

  def step(self, action: int) -> dm_env.TimeStep:
    """Steps up to action_repeat times and returns a post-processed step."""
    if self._reset_next_step:
      return self.reset()

    timestep_stack = []

    # Step on environment multiple times for each selected action.
    for _ in range(self._action_repeats):
      timestep = self._environment.step([np.array([action])])
      #print("timestep: {}".format(timestep.observation.shape))

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

    # print("timestep_stack: {}".format(timestep_stack))

    observation = self._observation_from_timestep_stack(timestep_stack)

    # print("observation_from_timestep_stack: {}".format(observation.shape))

    timestep = dm_env.TimeStep(
        step_type=step_type,
        reward=reward,
        observation=observation,
        discount=discount)
    
    return self._postprocess_observation(timestep)

  def _observation_from_timestep_stack(self,
                                       timestep_stack: List[dm_env.TimeStep]):
    """Compute the observation for a stack of timesteps."""
    # We use last timestep for lives only.
    
    
    observation = timestep_stack[-1].observation
    if self._num_stacked_frames == 1:
      pooled_obs = np.max(
        np.stack([
            s.observation
            for s in timestep_stack[-self._pooled_frames:]
        ]),
        axis=0)
    else:
      tempArray = []
      for s in timestep_stack:
        tempArray.append(s.observation)
      
      if len(tempArray) < self._num_stacked_frames : 
        for i in range (self._num_stacked_frames - len(tempArray)):
          zeroes_array = [0.0] * 128
          tempArray.append(zeroes_array)
      pooled_obs = np.stack(tempArray)
    
    return pooled_obs

  def _postprocess_ram(self, raw_pixels: np.ndarray):
    """Grayscale, cast and normalize the pooled pixel observations."""

    # Cache the raw i.e. un-(stacked|pooled|grayscaled|downscaled) observation.
    # This is useful for e.g. making videos.

    processed_pixels = raw_pixels
    processed_pixels = processed_pixels.astype(np.uint8, copy=False)
    cast_observation = processed_pixels

    stacked_observation = self._frame_stacker.step(cast_observation)

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
    return specs.Array(shape=(), dtype=np.float)

  @property
  def raw_observation(self) -> np.ndarray:
    """Returns the raw observation, after any pooling has been applied."""
    return self._raw_observation