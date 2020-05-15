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

"""Visual networks for processing pixel inputs."""

from typing import Union, Sequence, Callable
import sonnet as snt
import tensorflow as tf


class ResNetTorso(snt.Module):
  """ResNet architecture used in IMPALA paper."""

  def __init__(
      self,
      num_channels: Sequence[int] = (16, 32, 32),  # default to IMPALA resnet.
      num_blocks: Sequence[int] = (2, 2, 2),  # default to IMPALA resnet.
      num_output_hidden: Sequence[int] = (256,),  # default to IMPALA resnet.
      conv_shape: Union[int, Sequence[int]] = 3,
      conv_stride: Union[int, Sequence[int]] = 1,
      pool_size: Union[int, Sequence[int]] = 3,
      pool_stride: Union[int, Sequence[int]] = 2,
      data_format: str = 'NHWC',
      activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
      output_dtype: tf.DType = tf.float32,
      name: str = 'resnet_torso'):
    super().__init__(name=name)

    self._output_dtype = output_dtype
    self._num_layers = len(num_blocks)

    # Create sequence of residual blocks.
    blocks = []
    for i in range(self._num_layers):
      blocks.append(
          ResidualBlockGroup(
              num_blocks[i],
              num_channels[i],
              conv_shape,
              conv_stride,
              pool_size,
              pool_stride,
              data_format=data_format,
              activation=activation))

    # Create output layer.
    out_layer = snt.nets.MLP(num_output_hidden, activation=activation)

    # Compose blocks and final layer.
    self._resnet = snt.Sequential(
        blocks + [activation, snt.Flatten(), out_layer])

  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    """Evaluates the ResidualPixelCore."""

    # Convert to floats.
    preprocessed_inputs = _preprocess_inputs(inputs, self._output_dtype)
    torso_output = self._resnet(preprocessed_inputs)

    return torso_output


class ResidualBlockGroup(snt.Module):
  """Higher level block for ResNet implementation."""

  def __init__(self,
               num_blocks: int,
               num_output_channels: int,
               conv_shape: Union[int, Sequence[int]],
               conv_stride: Union[int, Sequence[int]],
               pool_shape: Union[int, Sequence[int]],
               pool_stride: Union[int, Sequence[int]],
               data_format: str = 'NHWC',
               activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
               name: str = None):
    super().__init__(name=name)

    self._num_blocks = num_blocks
    self._data_format = data_format
    self._activation = activation

    # The pooling operation expects a 2-rank shape/stride (height and width).
    if isinstance(pool_shape, int):
      pool_shape = 2 * [pool_shape]
    if isinstance(pool_stride, int):
      pool_stride = 2 * [pool_stride]

    # Create a Conv2D factory since we'll be making quite a few.
    def build_conv_layer(name: str):
      return snt.Conv2D(
          num_output_channels,
          conv_shape,
          stride=conv_stride,
          padding='SAME',
          data_format=data_format,
          name=name)

    # Create a pooling layer.
    def pooling_layer(inputs: tf.Tensor) -> tf.Tensor:
      return tf.nn.pool(
          inputs,
          pool_shape,
          pooling_type='MAX',
          strides=pool_stride,
          padding='SAME',
          data_format=data_format)

    # Create an initial conv layer and pooling to scale the image down.
    self._downscale = snt.Sequential(
        [build_conv_layer('downscale'), pooling_layer])

    # Residual block(s).
    self._convs = []
    for i in range(self._num_blocks):
      name = 'residual_block_%d' % i
      self._convs.append(
          [build_conv_layer(name + '_0'),
           build_conv_layer(name + '_1')])

  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    # Downscale the inputs.
    conv_out = self._downscale(inputs)

    # Apply (sequence of) residual block(s).
    for i in range(self._num_blocks):
      block_input = conv_out
      conv_out = self._activation(conv_out)
      conv_out = self._convs[i][0](conv_out)
      conv_out = self._activation(conv_out)
      conv_out = self._convs[i][1](conv_out)
      conv_out += block_input
    return conv_out


def _preprocess_inputs(inputs: tf.Tensor, output_dtype: tf.DType) -> tf.Tensor:
  """Returns the `Tensor` corresponding to the preprocessed inputs."""
  rank = inputs.shape.rank
  if rank < 4:
    raise ValueError(
        'Input Tensor must have at least 4 dimensions (for '
        'batch size, height, width, and channels), but it only has '
        '{}'.format(rank))

  flattened_inputs = snt.Flatten(preserve_dims=3)(inputs)
  processed_inputs = tf.image.convert_image_dtype(
      flattened_inputs, dtype=output_dtype)
  return processed_inputs
