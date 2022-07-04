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

"""Atari wrapper using Opencv for pixel prepocessing.

Note that the AtariWrapper comes with several options, including the legacy
style that allows reproducing behavior of previous Acme versions.

To reproduce accurate standard result, we recommend using the default
configuration.
"""

from typing import List

from acme.wrappers import atari_wrapper
# pytype: disable=import-error
import cv2
# pytype: enable=import-error
import dm_env
import numpy as np


class AtariWrapperDopamine(atari_wrapper.BaseAtariWrapper):
  """Atari wrapper that matches exactly Dopamine's prepocessing.

  Wraning: using this wrapper requires that you have opencv and its dependencies
  installed. In general, opencv is not required for Acme.
  """

  def _preprocess_pixels(self, timestep_stack: List[dm_env.TimeStep]):
    """Preprocess Atari frames."""

    # 1. RBG to grayscale
    def rgb_to_grayscale(obs):
      if self._grayscaling:
        return np.tensordot(obs, [0.2989, 0.5870, 0.1140], (-1, 0))
      return obs

    # 2. Max pooling
    processed_pixels = np.max(
        np.stack([
            rgb_to_grayscale(s.observation[atari_wrapper.RGB_INDEX])
            for s in timestep_stack[-self._pooled_frames:]
        ]),
        axis=0)

    # 3. Resize
    processed_pixels = np.round(processed_pixels).astype(np.uint8)
    if self._scale_dims != processed_pixels.shape[:2]:
      processed_pixels = cv2.resize(
          processed_pixels, (self._width, self._height),
          interpolation=cv2.INTER_AREA)

      processed_pixels = np.round(processed_pixels).astype(np.uint8)

    return processed_pixels
