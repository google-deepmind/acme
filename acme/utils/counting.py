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

"""A simple, hierarchical distributed counter."""

import threading
import time
from typing import Dict, Mapping, Optional, Union

from acme import core

Number = Union[int, float]


class Counter(core.Saveable):
  """A simple counter object that can periodically sync with a parent."""

  def __init__(self,
               parent: Optional['Counter'] = None,
               prefix: str = '',
               time_delta: float = 1.0):
    """Initialize the counter.

    Args:
      parent: a Counter object to cache locally (or None for no caching).
      prefix: string prefix to use for all local counts.
      time_delta: time difference in seconds between syncing with the parent
        counter.
    """

    self._parent = parent
    self._prefix = prefix
    self._time_delta = time_delta

    # Hold local counts and we'll lock around that.
    self._counts = {}
    self._lock = threading.Lock()

    # We'll sync the first time get_counts is called.
    self._cache = {}
    self._time = 0.0

  def increment(self, **counts: Number) -> Dict[str, Number]:
    """Increment a set of counters.

    Args:
      **counts: keyword arguments specifying count increments.

    Returns:
      The updated counts.
    """
    with self._lock:
      for key, value in counts.items():
        self._counts.setdefault(key, 0)
        self._counts[key] += value
    return self.get_counts()

  def get_counts(self) -> Dict[str, Number]:
    """Return all counts tracked by this counter."""
    now = time.time()
    # TODO(b/144421838): use futures instead of blocking.
    if self._parent and (now - self._time) > self._time_delta:
      with self._lock:
        counts = _prefix_keys(self._counts, self._prefix)
        self._counts = {}
      self._cache = self._parent.increment(**counts)
      self._time = now

    # Potentially prefix the keys in the counts dictionary.
    counts = _prefix_keys(self._counts, self._prefix)

    # If there's no prefix make a copy of the dictionary so we don't modify the
    # internal self._counts.
    if not self._prefix:
      counts = dict(counts)

    # Combine local counts with any parent counts.
    for key, value in self._cache.items():
      counts[key] = counts.get(key, 0) + value

    return counts

  def save(self) -> Mapping[str, Mapping[str, Number]]:
    return {'counts': self._counts, 'cache': self._cache}

  def restore(self, state: Mapping[str, Mapping[str, Number]]):
    # Force a sync, if necessary, on the next get_counts call.
    self._time = 0.
    self._counts = state['counts']
    self._cache = state['cache']


def _prefix_keys(dictionary: Dict[str, Number], prefix: str):
  """Return a dictionary with prefixed keys.

  Args:
    dictionary: dictionary to return a copy of.
    prefix: string to use as the prefix.

  Returns:
    Return a copy of the given dictionary whose keys are replaced by
    "{prefix}_{key}". If the prefix is the empty string it returns the given
    dictionary unchanged.
  """
  if prefix:
    dictionary = {f'{prefix}_{k}': v for k, v in dictionary.items()}
  return dictionary
