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

"""Networks used in continuous control."""

from typing import Sequence

from acme import types
from acme.tf import utils as tf2_utils
from acme.tf.networks import base
import sonnet as snt
import tensorflow as tf

uniform_initializer = tf.initializers.VarianceScaling(
    distribution='uniform', mode='fan_out', scale=0.333)


class NearZeroInitializedLinear(snt.Linear):
  """Simple linear layer, initialized at near zero weights and zero biases."""

  def __init__(self, output_size: int, scale: float = 1e-4):
    super().__init__(output_size, w_init=tf.initializers.VarianceScaling(scale))


class LayerNormMLP(snt.Module):
  """Simple feedforward MLP torso with initial layer-norm.

  This module is an MLP which uses LayerNorm (with a tanh normalizer) on the
  first layer and non-linearities (elu) on all but the last remaining layers.
  """

  def __init__(self, layer_sizes: Sequence[int], activate_final: bool = False):
    """Construct the MLP.

    Args:
      layer_sizes: a sequence of ints specifying the size of each layer.
      activate_final: whether or not to use the activation function on the final
        layer of the neural network.
    """
    super().__init__(name='feedforward_mlp_torso')

    self._network = snt.Sequential([
        snt.Linear(layer_sizes[0], w_init=uniform_initializer),
        snt.LayerNorm(
            axis=slice(1, None), create_scale=True, create_offset=True),
        tf.nn.tanh,
        snt.nets.MLP(
            layer_sizes[1:],
            w_init=uniform_initializer,
            activation=tf.nn.elu,
            activate_final=activate_final),
    ])

  def __call__(self, observations: types.Nest) -> tf.Tensor:
    """Forwards the policy network."""
    return self._network(tf2_utils.batch_concat(observations))


class ResidualLayernormWrapper(snt.Module):
  """Wrapper that applies residual connections and layer norm."""

  def __init__(self, layer: base.Module):
    """Creates the Wrapper Class.

    Args:
      layer: module to wrap.
    """

    super().__init__(name='ResidualLayernormWrapper')
    self._layer = layer

    self._layer_norm = snt.LayerNorm(
        axis=-1, create_scale=True, create_offset=True)

  def __call__(self, inputs: tf.Tensor):
    """Returns the result of the residual and layernorm computation.

    Args:
      inputs: inputs to the main module.
    """

    # Apply main module.
    outputs = self._layer(inputs)
    outputs = self._layer_norm(outputs + inputs)

    return outputs


class LayerNormAndResidualMLP(snt.Module):
  """MLP with residual connections and layer norm.

  An MLP which applies residual connection and layer normalisation every two
  linear layers. Similar to Resnet, but with FC layers instead of convolutions.
  """

  def __init__(self, hidden_size: int, num_blocks: int):
    """Create the model.

    Args:
      hidden_size: width of each hidden layer.
      num_blocks: number of blocks, each block being MLP([hidden_size,
        hidden_size]) + layer norm + residual connection.
    """
    super().__init__(name='LayerNormAndResidualMLP')

    # Create initial MLP layer.
    layers = [snt.nets.MLP([hidden_size], w_init=uniform_initializer)]

    # Follow it up with num_blocks MLPs with layernorm and residual connections.
    for _ in range(num_blocks):
      mlp = snt.nets.MLP([hidden_size, hidden_size], w_init=uniform_initializer)
      layers.append(ResidualLayernormWrapper(mlp))

    self._module = snt.Sequential(layers)

  def __call__(self, inputs: tf.Tensor):
    return self._module(inputs)
