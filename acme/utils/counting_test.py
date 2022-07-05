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

"""Tests for acme.utils.counting."""

import threading

from acme.utils import counting

from absl.testing import absltest


class Barrier:
  """Defines a simple barrier class to synchronize on a particular event."""

  def __init__(self, num_threads):
    """Constructor.

    Args:
      num_threads: int - how many threads will be syncronizing on this barrier
    """
    self._num_threads = num_threads
    self._count = 0
    self._cond = threading.Condition()

  def wait(self):
    """Waits on the barrier until all threads have called this method."""
    with self._cond:
      self._count += 1
      self._cond.notifyAll()
      while self._count < self._num_threads:
        self._cond.wait()


class CountingTest(absltest.TestCase):

  def test_counter_threading(self):
    counter = counting.Counter()
    num_threads = 10
    barrier = Barrier(num_threads)

    # Increment in every thread at the same time.
    def add_to_counter():
      barrier.wait()
      counter.increment(foo=1)

    # Run the threads.
    threads = []
    for _ in range(num_threads):
      t = threading.Thread(target=add_to_counter)
      t.start()
      threads.append(t)
    for t in threads:
      t.join()

    # Make sure the counter has been incremented once per thread.
    counts = counter.get_counts()
    self.assertEqual(counts['foo'], num_threads)

  def test_counter_caching(self):
    parent = counting.Counter()
    counter = counting.Counter(parent, time_delta=0.)
    counter.increment(foo=12)
    self.assertEqual(parent.get_counts(), counter.get_counts())

  def test_shared_counts(self):
    # Two counters with shared parent should share counts (modulo namespacing).
    parent = counting.Counter()
    child1 = counting.Counter(parent, 'child1')
    child2 = counting.Counter(parent, 'child2')
    child1.increment(foo=1)
    result = child2.increment(foo=2)
    expected = {'child1_foo': 1, 'child2_foo': 2}
    self.assertEqual(result, expected)

  def test_return_only_prefixed(self):
    parent = counting.Counter()
    child1 = counting.Counter(
        parent, 'child1', time_delta=0., return_only_prefixed=False)
    child2 = counting.Counter(
        parent, 'child2', time_delta=0., return_only_prefixed=True)
    child1.increment(foo=1)
    child2.increment(bar=1)
    self.assertEqual(child1.get_counts(), {'child1_foo': 1, 'child2_bar': 1})
    self.assertEqual(child2.get_counts(), {'bar': 1})

  def test_get_steps_key(self):
    parent = counting.Counter()
    child1 = counting.Counter(
        parent, 'child1', time_delta=0., return_only_prefixed=False)
    child2 = counting.Counter(
        parent, 'child2', time_delta=0., return_only_prefixed=True)
    self.assertEqual(child1.get_steps_key(), 'child1_steps')
    self.assertEqual(child2.get_steps_key(), 'steps')
    child1.increment(steps=1)
    child2.increment(steps=2)
    self.assertEqual(child1.get_counts().get(child1.get_steps_key()), 1)
    self.assertEqual(child2.get_counts().get(child2.get_steps_key()), 2)

  def test_parent_prefix(self):
    parent = counting.Counter(prefix='parent')
    child = counting.Counter(parent, prefix='child', time_delta=0.)
    self.assertEqual(child.get_steps_key(), 'child_steps')

if __name__ == '__main__':
  absltest.main()
