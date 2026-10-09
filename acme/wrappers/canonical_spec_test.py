# Copyright 2026 DeepMind Technologies Limited.
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

"""Tests for discrete and continuous CanonicalSpecWrapper action handling."""

from acme.wrappers import canonical_spec
import dm_env
from dm_env import specs
import numpy as np

from absl.testing import absltest


class _ActionEnvironment(dm_env.Environment):

  def __init__(self, action_spec):
    self._action_spec = action_spec
    self.last_action = None

  def action_spec(self):
    return self._action_spec

  def reset(self):
    return dm_env.restart(np.zeros((), dtype=np.float32))

  def step(self, action):
    self.last_action = action
    return dm_env.transition(
        reward=0.0, observation=np.zeros((), dtype=np.float32)
    )

  def observation_spec(self):
    return specs.Array((), np.float32)

  def reward_spec(self):
    return specs.Array((), np.float32)

  def discount_spec(self):
    return specs.BoundedArray((), np.float32, minimum=0.0, maximum=1.0)


class CanonicalSpecDiscreteTest(absltest.TestCase):

  def test_discrete_spec_is_not_reconstructed_with_bounded_kwargs(self):
    original = specs.DiscreteArray(num_values=5, dtype=np.int32, name='choice')
    wrapped = canonical_spec.CanonicalSpecWrapper(_ActionEnvironment(original))

    result = wrapped.action_spec()

    self.assertIs(result, original)
    self.assertEqual(result.num_values, 5)
    self.assertEqual(result.name, 'choice')
    np.testing.assert_array_equal(result.minimum, 0)
    np.testing.assert_array_equal(result.maximum, 4)

  def test_discrete_action_passes_through_without_scaling(self):
    environment = _ActionEnvironment(specs.DiscreteArray(5))
    wrapped = canonical_spec.CanonicalSpecWrapper(environment, clip=True)

    for value in (0, 2, 4):
      action = np.int32(value)
      wrapped.step(action)
      self.assertIs(environment.last_action, action)
      self.assertEqual(environment.last_action, value)

  def test_bounded_continuous_actions_are_still_canonicalized(self):
    original = specs.BoundedArray(
        (2,),
        np.float32,
        minimum=[-2.0, 10.0],
        maximum=[4.0, 14.0],
        name='control',
    )
    environment = _ActionEnvironment(original)
    wrapped = canonical_spec.CanonicalSpecWrapper(environment)

    converted = wrapped.action_spec()
    self.assertIsInstance(converted, specs.BoundedArray)
    self.assertIsNot(converted, original)
    np.testing.assert_array_equal(converted.minimum, [-1.0, -1.0])
    np.testing.assert_array_equal(converted.maximum, [1.0, 1.0])
    self.assertEqual(converted.name, 'control')
    wrapped.step(np.array([-1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(environment.last_action, [-2.0, 14.0])

  def test_nested_discrete_and_continuous_specs(self):
    environment = _ActionEnvironment(
        {
            'choice': specs.DiscreteArray(3),
            'control': specs.BoundedArray(
                (1,), np.float32, minimum=-10.0, maximum=10.0
            ),
            'unbounded': specs.Array((1,), np.float32),
        }
    )
    wrapped = canonical_spec.CanonicalSpecWrapper(environment, clip=True)

    converted = wrapped.action_spec()
    self.assertIs(converted['choice'], environment.action_spec()['choice'])
    self.assertIs(
        converted['unbounded'], environment.action_spec()['unbounded']
    )
    np.testing.assert_array_equal(converted['control'].minimum, [-1.0])

    wrapped.step(
        {
            'choice': np.int32(2),
            'control': np.array([2.0], dtype=np.float32),
            'unbounded': np.array([3.0], dtype=np.float32),
        }
    )
    self.assertEqual(environment.last_action['choice'], 2)
    np.testing.assert_allclose(environment.last_action['control'], [10.0])
    np.testing.assert_allclose(environment.last_action['unbounded'], [3.0])

  def test_discrete_spec_with_single_precision_conversion(self):
    from acme.wrappers import single_precision

    original = _ActionEnvironment(specs.DiscreteArray(7, dtype=np.int64))
    wrapped = canonical_spec.CanonicalSpecWrapper(
        single_precision.SinglePrecisionWrapper(original)
    )
    converted = wrapped.action_spec()
    self.assertIsInstance(converted, specs.DiscreteArray)
    self.assertEqual(converted.num_values, 7)
    wrapped.step(np.int32(6))
    self.assertEqual(original.last_action, 6)


if __name__ == '__main__':
  absltest.main()
