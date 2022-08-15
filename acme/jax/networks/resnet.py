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

"""ResNet Modules."""

import enum
import functools
from typing import Callable, Sequence, Union
import haiku as hk
import jax
import jax.numpy as jnp

InnerOp = Union[hk.Module, Callable[..., jnp.ndarray]]
MakeInnerOp = Callable[..., InnerOp]
NonLinearity = Callable[[jnp.ndarray], jnp.ndarray]


class ResidualBlock(hk.Module):
  """Residual block of operations, e.g. convolutional or MLP."""

  def __init__(self,
               make_inner_op: MakeInnerOp,
               non_linearity: NonLinearity = jax.nn.relu,
               use_layer_norm: bool = False,
               name: str = 'residual_block'):
    super().__init__(name=name)
    self.inner_op1 = make_inner_op()
    self.inner_op2 = make_inner_op()
    self.non_linearity = non_linearity
    self.use_layer_norm = use_layer_norm

    if use_layer_norm:
      self.layernorm1 = hk.LayerNorm(
          axis=(1, 2, 3), create_scale=True, create_offset=True, eps=1e-6)
      self.layernorm2 = hk.LayerNorm(
          axis=(1, 2, 3), create_scale=True, create_offset=True, eps=1e-6)

  def __call__(self, x: jnp.ndarray):
    output = x

    # First layer in residual block.
    if self.use_layer_norm:
      output = self.layernorm1(output)
    output = self.non_linearity(output)
    output = self.inner_op1(output)

    # Second layer in residual block.
    if self.use_layer_norm:
      output = self.layernorm2(output)
    output = self.non_linearity(output)
    output = self.inner_op2(output)
    return x + output


# TODO(nikola): Remove this enum and configure downsampling with a layer factory
# instead.
class DownsamplingStrategy(enum.Enum):
  AVG_POOL = 'avg_pool'
  CONV_MAX = 'conv+max'  # Used in IMPALA
  LAYERNORM_RELU_CONV = 'layernorm+relu+conv'  # Used in MuZero
  CONV = 'conv'


def make_downsampling_layer(
    strategy: Union[str, DownsamplingStrategy],
    output_channels: int,
) -> hk.SupportsCall:
  """Returns a sequence of modules corresponding to the desired downsampling."""
  strategy = DownsamplingStrategy(strategy)

  if strategy is DownsamplingStrategy.AVG_POOL:
    return hk.AvgPool(window_shape=(3, 3, 1), strides=(2, 2, 1), padding='SAME')

  elif strategy is DownsamplingStrategy.CONV:
    return hk.Sequential([
        hk.Conv2D(
            output_channels,
            kernel_shape=3,
            stride=2,
            w_init=hk.initializers.TruncatedNormal(1e-2)),
    ])

  elif strategy is DownsamplingStrategy.LAYERNORM_RELU_CONV:
    return hk.Sequential([
        hk.LayerNorm(
            axis=(1, 2, 3), create_scale=True, create_offset=True, eps=1e-6),
        jax.nn.relu,
        hk.Conv2D(
            output_channels,
            kernel_shape=3,
            stride=2,
            w_init=hk.initializers.TruncatedNormal(1e-2)),
    ])

  elif strategy is DownsamplingStrategy.CONV_MAX:
    return hk.Sequential([
        hk.Conv2D(output_channels, kernel_shape=3, stride=1),
        hk.MaxPool(window_shape=(3, 3, 1), strides=(2, 2, 1), padding='SAME')
    ])
  else:
    raise ValueError('Unrecognized downsampling strategy. Expected one of'
                     f' {[strategy.value for strategy in DownsamplingStrategy]}'
                     f' but received {strategy}.')


class ResNetTorso(hk.Module):
  """ResNetTorso for visual inputs, inspired by the IMPALA paper."""

  def __init__(self,
               channels_per_group: Sequence[int] = (16, 32, 32),
               blocks_per_group: Sequence[int] = (2, 2, 2),
               downsampling_strategies: Sequence[DownsamplingStrategy] = (
                   DownsamplingStrategy.CONV_MAX,) * 3,
               use_layer_norm: bool = False,
               name: str = 'resnet_torso'):
    super().__init__(name=name)
    self._channels_per_group = channels_per_group
    self._blocks_per_group = blocks_per_group
    self._downsampling_strategies = downsampling_strategies
    self._use_layer_norm = use_layer_norm

    if (len(channels_per_group) != len(blocks_per_group) or
        len(channels_per_group) != len(downsampling_strategies)):
      raise ValueError('Length of channels_per_group, blocks_per_group, and '
                       'downsampling_strategies must be equal. '
                       f'Got channels_per_group={channels_per_group}, '
                       f'blocks_per_group={blocks_per_group}, and'
                       f'downsampling_strategies={downsampling_strategies}.')

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    output = inputs
    channels_blocks_strategies = zip(self._channels_per_group,
                                     self._blocks_per_group,
                                     self._downsampling_strategies)

    for i, (num_channels, num_blocks,
            strategy) in enumerate(channels_blocks_strategies):
      output = make_downsampling_layer(strategy, num_channels)(output)

      for j in range(num_blocks):
        output = ResidualBlock(
            make_inner_op=functools.partial(
                hk.Conv2D, output_channels=num_channels, kernel_shape=3),
            use_layer_norm=self._use_layer_norm,
            name=f'residual_{i}_{j}')(
                output)

    return output
