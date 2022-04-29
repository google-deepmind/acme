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

"""Tests for action_metrics_observers."""


from acme import specs
from acme.testing import fakes
from acme.utils.observers import action_metrics
import dm_env
import numpy as np

from absl.testing import absltest


def _make_fake_env() -> dm_env.Environment:
  env_spec = specs.EnvironmentSpec(
      observations=specs.Array(shape=(10, 5), dtype=np.float32),
      actions=specs.BoundedArray(
          shape=(1,), dtype=np.float32, minimum=-100., maximum=100.),
      rewards=specs.Array(shape=(), dtype=np.float32),
      discounts=specs.BoundedArray(
          shape=(), dtype=np.float32, minimum=0., maximum=1.),
  )
  return fakes.Environment(env_spec, episode_length=10)

_FAKE_ENV = _make_fake_env()
_TIMESTEP = _FAKE_ENV.reset()


class ActionMetricsTest(absltest.TestCase):

  def test_observe_nothing(self):
    observer = action_metrics.ContinuousActionObserver()
    self.assertEqual({}, observer.get_metrics())

  def test_observe_first(self):
    observer = action_metrics.ContinuousActionObserver()
    observer.observe_first(env=_FAKE_ENV, timestep=_TIMESTEP)
    self.assertEqual({}, observer.get_metrics())

  def test_observe_single_step(self):
    observer = action_metrics.ContinuousActionObserver()
    observer.observe_first(env=_FAKE_ENV, timestep=_TIMESTEP)
    observer.observe(env=_FAKE_ENV, timestep=_TIMESTEP, action=np.array([1]))
    self.assertEqual(
        {
            'action[0]_max': 1,
            'action[0]_min': 1,
            'action[0]_mean': 1,
            'action[0]_p50': 1,
        },
        observer.get_metrics(),
    )

  def test_observe_multiple_step(self):
    observer = action_metrics.ContinuousActionObserver()
    observer.observe_first(env=_FAKE_ENV, timestep=_TIMESTEP)
    observer.observe(env=_FAKE_ENV, timestep=_TIMESTEP, action=np.array([1]))
    observer.observe(env=_FAKE_ENV, timestep=_TIMESTEP, action=np.array([4]))
    observer.observe(env=_FAKE_ENV, timestep=_TIMESTEP, action=np.array([5]))
    self.assertEqual(
        {
            'action[0]_max': 5,
            'action[0]_min': 1,
            'action[0]_mean': 10 / 3,
            'action[0]_p50': 4,
        },
        observer.get_metrics(),
    )

  def test_observe_zero_dimensions(self):
    observer = action_metrics.ContinuousActionObserver()
    observer.observe_first(env=_FAKE_ENV, timestep=_TIMESTEP)
    observer.observe(env=_FAKE_ENV, timestep=_TIMESTEP, action=np.array(1))
    self.assertEqual(
        {
            'action[]_max': 1,
            'action[]_min': 1,
            'action[]_mean': 1,
            'action[]_p50': 1,
        },
        observer.get_metrics(),
    )

  def test_observe_multiple_dimensions(self):
    observer = action_metrics.ContinuousActionObserver()
    observer.observe_first(env=_FAKE_ENV, timestep=_TIMESTEP)
    observer.observe(
        env=_FAKE_ENV, timestep=_TIMESTEP, action=np.array([[1, 2], [3, 4]]))
    np.testing.assert_equal(
        {
            'action[0, 0]_max': 1,
            'action[0, 0]_min': 1,
            'action[0, 0]_mean': 1,
            'action[0, 0]_p50': 1,
            'action[0, 1]_max': 2,
            'action[0, 1]_min': 2,
            'action[0, 1]_mean': 2,
            'action[0, 1]_p50': 2,
            'action[1, 0]_max': 3,
            'action[1, 0]_min': 3,
            'action[1, 0]_mean': 3,
            'action[1, 0]_p50': 3,
            'action[1, 1]_max': 4,
            'action[1, 1]_min': 4,
            'action[1, 1]_mean': 4,
            'action[1, 1]_p50': 4,
        },
        observer.get_metrics(),
    )


if __name__ == '__main__':
  absltest.main()

