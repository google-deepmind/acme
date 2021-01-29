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

"""A iterator that does zero-copy conversion of `tf.Tensor`s into `np.ndarray`s."""

from typing import Iterator

from acme import types
import numpy as np
import tree


class NumpyIterator(Iterator[types.NestedArray]):
  """Iterator over a dataset with elements converted to numpy.

  Note: This iterator returns read-only numpy arrays.

  This iterator (compared to `tf.data.Dataset.as_numpy_iterator()`) does not
  copy the data when comverting `tf.Tensor`s to `np.ndarray`s.

  TODO(b/178684359): Remove this when it is upstreamed into `tf.data`.
  """

  __slots__ = ['_iterator']

  def __init__(self, dataset):
    self._iterator: Iterator[types.NestedTensor] = iter(dataset)

  def __iter__(self) -> 'NumpyIterator':
    return self

  def __next__(self) -> types.NestedArray:
    return tree.map_structure(lambda t: np.asarray(memoryview(t)),
                              next(self._iterator))

  def next(self):
    return self.__next__()
