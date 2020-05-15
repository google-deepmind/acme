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

"""Tests for acme.utils.loggers."""

from absl.testing import absltest
from acme.utils.loggers import terminal


class LoggingTest(absltest.TestCase):

  def test_logging_output_format(self):
    inputs = {
        'c': 'foo',
        'a': 1337,
        'b': 42.0001,
    }
    expected_outputs = 'A = 1337 | B = 42.000 | C = foo'
    test_fn = lambda outputs: self.assertEqual(outputs, expected_outputs)

    logger = terminal.TerminalLogger(print_fn=test_fn)
    logger.write(inputs)

  def test_label(self):
    inputs = {'foo': 'bar', 'baz': 123}
    expected_outputs = '[Test] Baz = 123 | Foo = bar'
    test_fn = lambda outputs: self.assertEqual(outputs, expected_outputs)

    logger = terminal.TerminalLogger(print_fn=test_fn, label='test')
    logger.write(inputs)


if __name__ == '__main__':
  absltest.main()
