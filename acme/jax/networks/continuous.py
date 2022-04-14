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

from typing import Callable, Sequence

import haiku as hk
import jax
import jax.numpy as jnp

uniform_initializer = hk.initializers.UniformScaling(scale=0.333)


class NearZeroInitializedLinear(hk.Linear):
  """Simple linear layer, initialized at near zero weights and zero biases."""

  def __init__(self, output_size: int, scale: float = 1e-4):
    super().__init__(output_size, w_init=hk.initializers.VarianceScaling(scale))


class LayerNormMLP(hk.Module):
  """Simple feedforward MLP torso with initial layer-norm.

  This MLP's first linear layer is followed by a LayerNorm layer and a tanh
  non-linearity; subsequent layers use `activation`, which defaults to elu.

  Note! The default activation differs from the usual MLP default of ReLU for
  legacy reasons.
  """

  def __init__(self,
               layer_sizes: Sequence[int],
               w_init: hk.initializers.Initializer = uniform_initializer,
               activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.elu,
               activate_final: bool = False,
               name: str = 'feedforward_mlp_torso'):
    """Construct the MLP.

    Args:
      layer_sizes: a sequence of ints specifying the size of each layer.
      w_init: initializer for Linear layers.
      activation: nonlinearity to use in the MLP, defaults to elu.
        Note! The default activation differs from the usual MLP default of ReLU
        for legacy reasons.
      activate_final: whether or not to use the activation function on the final
        layer of the neural network.
      name: a name for the module.
    """
    super().__init__(name=name)

    self._network = hk.Sequential([
        hk.Linear(layer_sizes[0], w_init=w_init),
        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
        jax.lax.tanh,
        hk.nets.MLP(
            layer_sizes[1:],
            w_init=w_init,
            activation=activation,
            activate_final=activate_final),
    ])

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Forwards the policy network."""
    return self._network(inputs)
