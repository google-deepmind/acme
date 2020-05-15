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


class EnvironmentLoopTest(absltest.TestCase):

  def test_environment_loop(self):
    # Create the actor/environment and stick them in a loop.
    environment = fakes.DiscreteEnvironment(episode_length=10)
    actor = fakes.Actor(specs.make_environment_spec(environment))
    loop = environment_loop.EnvironmentLoop(environment, actor)

    # Run the loop. There should be episode_length+1 update calls per episode.
    loop.run(num_episodes=10)
    self.assertEqual(actor.num_updates, 100)


if __name__ == '__main__':
  absltest.main()
