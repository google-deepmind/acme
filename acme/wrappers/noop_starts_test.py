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

"""Tests for the noop starts wrapper."""

from unittest import mock

from acme import wrappers
from acme.testing import fakes
from dm_env import specs
import numpy as np

from absl.testing import absltest


class NoopStartsTest(absltest.TestCase):

  def test_reset(self):
    """Ensure that noop starts `reset` steps the environment multiple times."""
    noop_action = 0
    noop_max = 10
    seed = 24

    base_env = fakes.DiscreteEnvironment(
        action_dtype=np.int64,
        obs_dtype=np.int64,
        reward_spec=specs.Array(dtype=np.float64, shape=()))
    mock_step_fn = mock.MagicMock()
    expected_num_step_calls = np.random.RandomState(seed).randint(noop_max + 1)

    with mock.patch.object(base_env, 'step', mock_step_fn):
      env = wrappers.NoopStartsWrapper(
          base_env,
          noop_action=noop_action,
          noop_max=noop_max,
          seed=seed,
      )
      env.reset()

      # Test environment step called with noop action as part of wrapper.reset
      mock_step_fn.assert_called_with(noop_action)
      self.assertEqual(mock_step_fn.call_count, expected_num_step_calls)
      self.assertEqual(mock_step_fn.call_args, ((noop_action,), {}))

  def test_raises_value_error(self):
    """Ensure that wrapper raises error if noop_max is <0."""
    base_env = fakes.DiscreteEnvironment(
        action_dtype=np.int64,
        obs_dtype=np.int64,
        reward_spec=specs.Array(dtype=np.float64, shape=()))

    with self.assertRaises(ValueError):
      wrappers.NoopStartsWrapper(base_env, noop_action=0, noop_max=-1, seed=24)


if __name__ == '__main__':
  absltest.main()
