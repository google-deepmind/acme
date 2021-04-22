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

"""Tests for acme launchpad utilities."""

from absl.testing import absltest

from acme.utils import lp_utils


class EnvironmentLoopTest(absltest.TestCase):

  def test_partial_kwargs(self):

    def foo(a, b, c=2):
      return a, b, c

    def bar(a, b):
      return a, b

    # Override the default values. The last two should be no-ops.
    foo1 = lp_utils.partial_kwargs(foo, c=1)
    foo2 = lp_utils.partial_kwargs(foo)
    bar1 = lp_utils.partial_kwargs(bar)

    # Check that we raise errors on overriding kwargs with no default values
    with self.assertRaises(ValueError):
      lp_utils.partial_kwargs(foo, a=2)

    # CHeck the we raise if we try to override a kwarg that doesn't exist.
    with self.assertRaises(ValueError):
      lp_utils.partial_kwargs(foo, d=2)

    # Make sure we get back the correct values.
    self.assertEqual(foo1(1, 2), (1, 2, 1))
    self.assertEqual(foo2(1, 2), (1, 2, 2))
    self.assertEqual(bar1(1, 2), (1, 2))


if __name__ == '__main__':
  absltest.main()
