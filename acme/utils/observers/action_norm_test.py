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

"""Tests for acme.utils.observers.action_norm."""

from acme import specs
from acme.testing import fakes
from acme.utils.observers import action_norm
import dm_env
import numpy as np

from absl.testing import absltest


def _make_fake_env() -> dm_env.Environment:
  env_spec = specs.EnvironmentSpec(
      observations=specs.Array(shape=(10, 5), dtype=np.float32),
      actions=specs.BoundedArray(
          shape=(1,), dtype=np.float32, minimum=-10., maximum=10.),
      rewards=specs.Array(shape=(), dtype=np.float32),
      discounts=specs.BoundedArray(
          shape=(), dtype=np.float32, minimum=0., maximum=1.),
  )
  return fakes.Environment(env_spec, episode_length=10)


class ActionNormTest(absltest.TestCase):

  def test_basic(self):
    env = _make_fake_env()
    observer = action_norm.ActionNormObserver()
    timestep = env.reset()
    observer.observe_first(env, timestep)
    for it in range(5):
      action = np.ones((1,), dtype=np.float32) * it
      timestep = env.step(action)
      observer.observe(env, timestep, action)
    metrics = observer.get_metrics()
    self.assertLen(metrics, 3)
    np.testing.assert_equal(metrics['action_norm_min'], 0)
    np.testing.assert_equal(metrics['action_norm_max'], 4)
    np.testing.assert_equal(metrics['action_norm_avg'], 2)


if __name__ == '__main__':
  absltest.main()
