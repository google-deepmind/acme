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

"""Tests for the environment loop."""

from absl.testing import absltest

from acme import environment_loop
from acme import specs
from acme.testing import fakes

EPISODE_LENGTH = 10


class EnvironmentLoopTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Create the actor/environment and stick them in a loop.
    environment = fakes.DiscreteEnvironment(episode_length=EPISODE_LENGTH)
    self.actor = fakes.Actor(specs.make_environment_spec(environment))
    self.loop = environment_loop.EnvironmentLoop(environment, self.actor)

  def test_one_episode(self):
    result = self.loop.run_episode()
    self.assertDictContainsSubset({'episode_length': EPISODE_LENGTH}, result)
    self.assertIn('episode_return', result)
    self.assertIn('steps_per_second', result)

  def test_run_episodes(self):
    # Run the loop. There should be EPISODE_LENGTH update calls per episode.
    self.loop.run(num_episodes=10)
    self.assertEqual(self.actor.num_updates, 10 * EPISODE_LENGTH)

  def test_run_steps(self):
    # Run the loop. This will run 2 episodes so that total number of steps is
    # at least 15.
    self.loop.run(num_steps=EPISODE_LENGTH + 5)
    self.assertEqual(self.actor.num_updates, 2 * EPISODE_LENGTH)


if __name__ == '__main__':
  absltest.main()
