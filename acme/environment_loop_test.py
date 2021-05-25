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

from typing import Optional
from absl.testing import absltest
from absl.testing import parameterized

from acme import environment_loop
from acme import specs
from acme import types
from acme.testing import fakes
import numpy as np

EPISODE_LENGTH = 10

# Discount specs
F32_2_MIN_0_MAX_1 = specs.BoundedArray(
    dtype=np.float32, shape=(2,), minimum=0.0, maximum=1.0)
F32_2x1_MIN_0_MAX_1 = specs.BoundedArray(
    dtype=np.float32, shape=(2, 1), minimum=0.0, maximum=1.0)
TREE_MIN_0_MAX_1 = {'a': F32_2_MIN_0_MAX_1, 'b': F32_2x1_MIN_0_MAX_1}

# Reward specs
F32 = specs.Array(dtype=np.float32, shape=())
F32_1x3 = specs.Array(dtype=np.float32, shape=(1, 3))
TREE = {'a': F32, 'b': F32_1x3}

TEST_CASES = (
    ('scalar_discount_scalar_reward', None, None),
    ('vector_discount_scalar_reward', F32_2_MIN_0_MAX_1, F32),
    ('matrix_discount_matrix_reward', F32_2x1_MIN_0_MAX_1, F32_1x3),
    ('tree_discount_tree_reward', TREE_MIN_0_MAX_1, TREE),
    )


class EnvironmentLoopTest(parameterized.TestCase):

  @parameterized.named_parameters(*TEST_CASES)
  def test_one_episode(self, discount_spec, reward_spec):
    _, loop = _parameterized_setup(discount_spec, reward_spec)
    result = loop.run_episode()
    self.assertIn('episode_length', result)
    self.assertEqual(EPISODE_LENGTH, result['episode_length'])
    self.assertIn('episode_return', result)
    self.assertIn('steps_per_second', result)

  @parameterized.named_parameters(*TEST_CASES)
  def test_run_episodes(self, discount_spec, reward_spec):
    actor, loop = _parameterized_setup(discount_spec, reward_spec)

    # Run the loop. There should be EPISODE_LENGTH update calls per episode.
    loop.run(num_episodes=10)
    self.assertEqual(actor.num_updates, 10 * EPISODE_LENGTH)

  @parameterized.named_parameters(*TEST_CASES)
  def test_run_steps(self, discount_spec, reward_spec):
    actor, loop = _parameterized_setup(discount_spec, reward_spec)

    # Run the loop. This will run 2 episodes so that total number of steps is
    # at least 15.
    loop.run(num_steps=EPISODE_LENGTH + 5)
    self.assertEqual(actor.num_updates, 2 * EPISODE_LENGTH)


def _parameterized_setup(discount_spec: Optional[types.NestedSpec] = None,
                         reward_spec: Optional[types.NestedSpec] = None):
  """Common setup code that, unlike self.setUp, takes arguments.

  Args:
    discount_spec: None, or a (nested) specs.BoundedArray.
    reward_spec: None, or a (nested) specs.Array.
  Returns:
    environment, actor, loop
  """
  env_kwargs = {'episode_length': EPISODE_LENGTH}
  if discount_spec:
    env_kwargs['discount_spec'] = discount_spec
  if reward_spec:
    env_kwargs['reward_spec'] = reward_spec

  environment = fakes.DiscreteEnvironment(**env_kwargs)
  actor = fakes.Actor(specs.make_environment_spec(environment))
  loop = environment_loop.EnvironmentLoop(environment, actor)
  return actor, loop


if __name__ == '__main__':
  absltest.main()
