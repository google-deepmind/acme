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

import dataclasses

from acme import specs
from jax import lax
import jax.numpy as jnp


@dataclasses.dataclass
class ClipToSpec:
  """Clips inputs to within a BoundedArraySpec."""
  spec: specs.BoundedArray

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(inputs, self.spec.minimum, self.spec.maximum)


@dataclasses.dataclass
class RescaleToSpec:
  """Rescales inputs in [-1, 1] to match a BoundedArraySpec."""
  spec: specs.BoundedArray

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    scale = self.spec.maximum - self.spec.minimum
    offset = self.spec.minimum
    inputs = 0.5 * (inputs + 1.0)  # [0, 1]
    output = inputs * scale + offset  # [minimum, maximum]
    return output


@dataclasses.dataclass
class TanhToSpec:
  """Squashes real-valued inputs to match a BoundedArraySpec."""
  spec: specs.BoundedArray

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    scale = self.spec.maximum - self.spec.minimum
    offset = self.spec.minimum
    inputs = lax.tanh(inputs)  # [-1, 1]
    inputs = 0.5 * (inputs + 1.0)  # [0, 1]
    output = inputs * scale + offset  # [minimum, maximum]
    return output
