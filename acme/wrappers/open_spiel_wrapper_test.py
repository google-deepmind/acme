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

"""Tests for open_spiel_wrapper."""

import unittest

from dm_env import specs
import numpy as np

from absl.testing import absltest

SKIP_OPEN_SPIEL_TESTS = False
SKIP_OPEN_SPIEL_MESSAGE = 'open_spiel not installed.'

try:
  # pylint: disable=g-import-not-at-top
  # pytype: disable=import-error
  from acme.wrappers import open_spiel_wrapper
  from open_spiel.python import rl_environment
  # pytype: enable=import-error
except ModuleNotFoundError:
  SKIP_OPEN_SPIEL_TESTS = True


@unittest.skipIf(SKIP_OPEN_SPIEL_TESTS, SKIP_OPEN_SPIEL_MESSAGE)
class OpenSpielWrapperTest(absltest.TestCase):

  def test_tic_tac_toe(self):
    raw_env = rl_environment.Environment('tic_tac_toe')
    env = open_spiel_wrapper.OpenSpielWrapper(raw_env)

    # Test converted observation spec.
    observation_spec = env.observation_spec()
    self.assertEqual(type(observation_spec), open_spiel_wrapper.OLT)
    self.assertEqual(type(observation_spec.observation), specs.Array)
    self.assertEqual(type(observation_spec.legal_actions), specs.Array)
    self.assertEqual(type(observation_spec.terminal), specs.Array)

    # Test converted action spec.
    action_spec: specs.DiscreteArray = env.action_spec()
    self.assertEqual(type(action_spec), specs.DiscreteArray)
    self.assertEqual(action_spec.shape, ())
    self.assertEqual(action_spec.minimum, 0)
    self.assertEqual(action_spec.maximum, 8)
    self.assertEqual(action_spec.num_values, 9)
    self.assertEqual(action_spec.dtype, np.dtype('int32'))

    # Test step.
    timestep = env.reset()
    self.assertTrue(timestep.first())
    _ = env.step([0])
    env.close()


if __name__ == '__main__':
  absltest.main()
