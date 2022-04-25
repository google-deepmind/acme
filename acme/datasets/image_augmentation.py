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

"""Transformations to be applied to replay datasets for augmentation purposes."""

import enum

from acme import types
from acme.datasets import reverb as reverb_dataset
import reverb
import tensorflow as tf


class CropType(enum.Enum):
  """Types of cropping supported by the image aumentation transforms.

  BILINEAR: Continuously randomly located then bilinearly interpolated.
  ALIGNED: Aligned with input image's pixel grid.
  """
  BILINEAR = 'bilinear'
  ALIGNED = 'aligned'


def pad_and_crop(img: tf.Tensor,
                 pad_size: int = 4,
                 method: CropType = CropType.ALIGNED) -> tf.Tensor:
  """Pad and crop image to mimic a random translation with mirroring at edges.

  This implements the image augmentation from section 3.1 in (Kostrikov et al.)
  https://arxiv.org/abs/2004.13649.

  Args:
    img: The image to pad and crop. Its dimensions are [..., H, W, C] where ...
      are batch dimensions (if it has any).
    pad_size: The amount of padding to apply to the image before cropping it.
    method: The method to use for cropping the image, see `CropType` for
      details.

  Returns:
    The image after having been padded and cropped.
  """
  num_batch_dims = img.shape[:-3].rank

  if img.shape.is_fully_defined():
    img_shape = img.shape.as_list()
  else:
    img_shape = tf.shape(img)

  # Set paddings for height and width only, batches and channels set to [0, 0].
  paddings = [[0, 0]] * num_batch_dims  # Do not pad batch dims.
  paddings.extend([[pad_size, pad_size], [pad_size, pad_size], [0, 0]])

  # Pad using symmetric padding.
  padded_img = tf.pad(img, paddings=paddings, mode='SYMMETRIC')

  # Crop padded image using requested method.
  if method == CropType.ALIGNED:
    cropped_img = tf.image.random_crop(padded_img, img_shape)
  elif method == CropType.BILINEAR:
    height, width = img_shape[-3:-1]
    padded_height, padded_width = height + 2 * pad_size, width + 2 * pad_size

    # Pick a top-left point uniformly at random.
    top_left = tf.random.uniform(
        shape=(2,), maxval=2 * pad_size + 1, dtype=tf.int32)

    # This single box is applied to the entire batch if a batch is passed.
    batch_size = tf.shape(padded_img)[0]
    box = tf.cast(
        tf.tile(
            tf.expand_dims([
                top_left[0] / padded_height,
                top_left[1] / padded_width,
                (top_left[0] + height) / padded_height,
                (top_left[1] + width) / padded_width,
            ], axis=0), [batch_size, 1]),
        tf.float32)  # Shape [batch_size, 2].

    # Crop and resize according to `box` then reshape back to input shape.
    cropped_img = tf.image.crop_and_resize(
        padded_img,
        box,
        tf.range(batch_size),
        (height, width),
        method='bilinear')
    cropped_img = tf.reshape(cropped_img, img_shape)

  return cropped_img


def make_transform(
    observation_transform: types.TensorTransformation,
    transform_next_observation: bool = True,
) -> reverb_dataset.Transform:
  """Creates the appropriate dataset transform for the given signature."""

  if transform_next_observation:
    def transform(x: reverb.ReplaySample) -> reverb.ReplaySample:
      return x._replace(
          data=x.data._replace(
              observation=observation_transform(x.data.observation),
              next_observation=observation_transform(x.data.next_observation)))
  else:
    def transform(x: reverb.ReplaySample) -> reverb.ReplaySample:
      return x._replace(
          data=x.data._replace(
              observation=observation_transform(x.data.observation)))

  return transform
