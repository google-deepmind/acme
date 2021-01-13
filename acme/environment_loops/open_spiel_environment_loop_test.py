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

"""Tests for OpenSpiel environment loop."""

import unittest
from absl.testing import absltest
from absl.testing import parameterized

import acme
from acme import core
from acme import specs
from acme import types
from acme import wrappers

import dm_env
import numpy as np
import tree

SKIP_OPEN_SPIEL_TESTS = False
SKIP_OPEN_SPIEL_MESSAGE = 'open_spiel not installed.'

try:
  # pylint: disable=g-import-not-at-top
  # pytype: disable=import-error
  from acme.environment_loops import open_spiel_environment_loop
  from acme.wrappers import open_spiel_wrapper
  from open_spiel.python import rl_environment
  # pytype: disable=import-error

  class RandomActor(core.Actor):
    """Fake actor which generates random actions and validates specs."""

    def __init__(self, spec: specs.EnvironmentSpec):
      self._spec = spec
      self.num_updates = 0

    def select_action(self, observation: open_spiel_wrapper.OLT) -> int:
      _validate_spec(self._spec.observations, observation)
      legals = np.array(np.nonzero(observation.legal_actions), dtype=np.int32)
      return np.random.choice(legals[0])

    def observe_first(self, timestep: dm_env.TimeStep):
      _validate_spec(self._spec.observations, timestep.observation)

    def observe(self, action: types.NestedArray,
                next_timestep: dm_env.TimeStep):
      _validate_spec(self._spec.actions, action)
      _validate_spec(self._spec.rewards, next_timestep.reward)
      _validate_spec(self._spec.discounts, next_timestep.discount)
      _validate_spec(self._spec.observations, next_timestep.observation)

    def update(self, wait: bool = False):
      self.num_updates += 1

except ModuleNotFoundError:
  SKIP_OPEN_SPIEL_TESTS = True


def _validate_spec(spec: types.NestedSpec, value: types.NestedArray):
  """Validate a value from a potentially nested spec."""
  tree.assert_same_structure(value, spec)
  tree.map_structure(lambda s, v: s.validate(v), spec, value)


@unittest.skipIf(SKIP_OPEN_SPIEL_TESTS, SKIP_OPEN_SPIEL_MESSAGE)
class OpenSpielEnvironmentLoopTest(parameterized.TestCase):

  def test_loop_run(self):
    raw_env = rl_environment.Environment('tic_tac_toe')
    env = open_spiel_wrapper.OpenSpielWrapper(raw_env)
    env = wrappers.SinglePrecisionWrapper(env)
    environment_spec = acme.make_environment_spec(env)

    actors = []
    for _ in range(env.num_players):
      actors.append(RandomActor(environment_spec))

    loop = open_spiel_environment_loop.OpenSpielEnvironmentLoop(env, actors)
    result = loop.run_episode()
    self.assertIn('episode_length', result)
    self.assertIn('episode_return', result)
    self.assertIn('steps_per_second', result)

    loop.run(num_episodes=10)
    loop.run(num_steps=100)


if __name__ == '__main__':
  absltest.main()
