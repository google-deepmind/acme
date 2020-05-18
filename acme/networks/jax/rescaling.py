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

"""Rescaling layers (e.g. to match action specs)."""

from acme import specs
import haiku as hk
from jax import lax
import jax.numpy as jnp


class ClipToSpec(hk.Module):
  """Sonnet module clipping inputs to within a BoundedArraySpec."""

  def __init__(self, spec: specs.BoundedArray, name: str = 'clip_to_spec'):
    super().__init__(name=name)
    self._min = spec.minimum
    self._max = spec.maximum

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(inputs, self._min, self._max)


class RescaleToSpec(hk.Module):
  """Sonnet module rescaling inputs in [-1, 1] to match a BoundedArraySpec."""

  def __init__(self, spec: specs.BoundedArray, name: str = 'rescale_to_spec'):
    super().__init__(name=name)
    self._scale = spec.maximum - spec.minimum
    self._offset = spec.minimum

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    inputs = 0.5 * (inputs + 1.0)  # [0, 1]
    output = inputs * self._scale + self._offset  # [minimum, maximum]

    return output


class TanhToSpec(hk.Module):
  """Sonnet module squashing real-valued inputs to match a BoundedArraySpec."""

  def __init__(self, spec: specs.BoundedArray, name: str = 'tanh_to_spec'):
    super().__init__(name=name)
    self._scale = spec.maximum - spec.minimum
    self._offset = spec.minimum

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    inputs = lax.tanh(inputs)  # [-1, 1]
    inputs = 0.5 * (inputs + 1.0)  # [0, 1]
    output = inputs * self._scale + self._offset  # [minimum, maximum]
    return output
