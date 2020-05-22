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

"""Base interfaces for networks."""

import abc
from typing import Any, Callable, Tuple

from acme import types
import haiku as hk
import jax.numpy as jnp

# Commonly-used types.
QValues = jnp.ndarray
Logits = jnp.ndarray
Value = jnp.ndarray

# Commonly-used function/network signatures.
QNetwork = Callable[[types.NestedArray], QValues]
RNNState = Any
PolicyValueRNN = Callable[[types.NestedArray, RNNState],
                          Tuple[Tuple[Logits, Value], RNNState]]


class Module(hk.Module, abc.ABC):
  """A base class for module with abstract __call__ method."""

  @abc.abstractmethod
  def __call__(self, *args, **kwargs):
    """Forward pass of the module."""
