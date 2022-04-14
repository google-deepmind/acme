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

"""Iterator utilities."""
import itertools
import operator
from typing import Any, Iterator, List, Sequence


def unzip_iterators(zipped_iterators: Iterator[Sequence[Any]],
                    num_sub_iterators: int) -> List[Iterator[Any]]:
  """Returns unzipped iterators.

  Note that simply returning:
    [(x[i] for x in iter_tuple[i]) for i in range(num_sub_iterators)]
  seems to cause all iterators to point to the final value of i, thus causing
  all sub_learners to consume data from this final iterator.

  Args:
    zipped_iterators: zipped iterators (e.g., from zip_iterators()).
    num_sub_iterators: the number of sub-iterators in the zipped iterator.
  """
  iter_tuple = itertools.tee(zipped_iterators, num_sub_iterators)
  return [
      map(operator.itemgetter(i), iter_tuple[i])
      for i in range(num_sub_iterators)
  ]
