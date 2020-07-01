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

"""Tensor framework-agnostic utilities for manipulating nested structures."""

from typing import Iterable, List, TypeVar

import numpy as np
import tree

ElementType = TypeVar('ElementType')


def fast_map_structure(func, *structure):
  """Faster map_structure implementation which skips some error checking."""
  flat_structure = (tree.flatten(s) for s in structure)
  entries = zip(*flat_structure)
  # Arbitrarily choose one of the structures of the original sequence (the last)
  # to match the structure for the flattened sequence.
  return tree.unflatten_as(structure[-1], [func(*x) for x in entries])


def stack_sequence_fields(sequence: Iterable[ElementType]) -> ElementType:
  """Stacks a list of identically nested objects.

  This takes a sequence of identically nested objects and returns a single
  nested object whose ith leaf is a stacked numpy array of the corresponding
  ith leaf from each element of the sequence.

  For example, if `sequence` is:

  ```python
  [{
        'action': np.array([1.0]),
        'observation': (np.array([0.0, 1.0, 2.0]),),
        'reward': 1.0
   }, {
        'action': np.array([0.5]),
        'observation': (np.array([1.0, 2.0, 3.0]),),
        'reward': 0.0
   }, {
        'action': np.array([0.3]),1
        'observation': (np.array([2.0, 3.0, 4.0]),),
        'reward': 0.5
   }]
  ```

  Then this function will return:

  ```python
  {
      'action': np.array([....])         # array shape = [3 x 1]
      'observation': (np.array([...]),)  # array shape = [3 x 3]
      'reward': np.array([...])          # array shape = [3]
  }
  ```

  Note that the 'observation' entry in the above example has two levels of
  nesting, i.e it is a tuple of arrays.

  Args:
    sequence: a list of identically nested objects.

  Returns:
    A nested object with numpy.

  Raises:
    ValueError: If `sequence` is an empty sequence.
  """
  # Handle empty input sequences.
  if not sequence:
    raise ValueError('Input sequence must not be empty')

  return fast_map_structure(lambda *values: np.asarray(values), *sequence)


def unstack_sequence_fields(struct: ElementType,
                            batch_size: int) -> List[ElementType]:
  """Converts a struct of batched arrays to a list of structs.

  This is effectively the inverse of `stack_sequence_fields`.

  Args:
    struct: An (arbitrarily nested) structure of arrays.
    batch_size: The length of the leading dimension of each array in the struct.
      This is assumed to be static and known.

  Returns:
    A list of structs with the same structure as `struct`, where each leaf node
     is an unbatched element of the original leaf node.
  """

  return [
      tree.map_structure(lambda s, i=i: s[i], struct) for i in range(batch_size)
  ]
