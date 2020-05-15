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

"""Common types used throughout Acme."""

from typing import Any, Callable, Iterable, Mapping, Union
from acme import specs

# Define types for nested arrays and tensors.
# TODO(b/144758674): Replace these with recursive type definitions.
NestedArray = Any
NestedTensor = Any

NestedSpec = Union[
    specs.Array,
    Iterable['NestedSpec'],
    Mapping[Any, 'NestedSpec'],  # pytype: disable=not-supported-yet
]

# TODO(b/144763593): Replace all instances of nest with the tensor/array types.
Nest = Union[NestedArray, NestedTensor, NestedSpec]

TensorTransformation = Callable[[NestedTensor], NestedTensor]
