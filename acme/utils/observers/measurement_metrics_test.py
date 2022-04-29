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

"""Tests for measurement_metrics."""

import copy
from unittest import mock

from acme import specs
from acme.testing import fakes
from acme.utils.observers import measurement_metrics
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
_TIMESTEP = mock.MagicMock(spec=dm_env.TimeStep)

_TIMESTEP.observation = [1.0, -2.0]


class MeasurementMetricsTest(absltest.TestCase):

  def test_observe_nothing(self):
    observer = measurement_metrics.MeasurementObserver()
    self.assertEqual({}, observer.get_metrics())

  def test_observe_first(self):
    observer = measurement_metrics.MeasurementObserver()
    observer.observe_first(env=_FAKE_ENV, timestep=_TIMESTEP)
    self.assertEqual({}, observer.get_metrics())

  def test_observe_single_step(self):
    observer = measurement_metrics.MeasurementObserver()
    observer.observe_first(env=_FAKE_ENV, timestep=_TIMESTEP)
    observer.observe(env=_FAKE_ENV, timestep=_TIMESTEP, action=np.array([1]))
    self.assertEqual(
        {
            'measurement[0]_max': 1.0,
            'measurement[0]_mean': 1.0,
            'measurement[0]_p25': 1.0,
            'measurement[0]_p50': 1.0,
            'measurement[0]_p75': 1.0,
            'measurement[1]_max': -2.0,
            'measurement[1]_mean': -2.0,
            'measurement[1]_p25': -2.0,
            'measurement[1]_p50': -2.0,
            'measurement[1]_p75': -2.0,
            'measurement[0]_min': 1.0,
            'measurement[1]_min': -2.0,
        },
        observer.get_metrics(),
    )

  def test_observe_multiple_step_same_observation(self):
    observer = measurement_metrics.MeasurementObserver()
    observer.observe_first(env=_FAKE_ENV, timestep=_TIMESTEP)
    observer.observe(env=_FAKE_ENV, timestep=_TIMESTEP, action=np.array([1]))
    observer.observe(env=_FAKE_ENV, timestep=_TIMESTEP, action=np.array([4]))
    observer.observe(env=_FAKE_ENV, timestep=_TIMESTEP, action=np.array([5]))
    self.assertEqual(
        {
            'measurement[0]_max': 1.0,
            'measurement[0]_mean': 1.0,
            'measurement[0]_p25': 1.0,
            'measurement[0]_p50': 1.0,
            'measurement[0]_p75': 1.0,
            'measurement[1]_max': -2.0,
            'measurement[1]_mean': -2.0,
            'measurement[1]_p25': -2.0,
            'measurement[1]_p50': -2.0,
            'measurement[1]_p75': -2.0,
            'measurement[0]_min': 1.0,
            'measurement[1]_min': -2.0,
        },
        observer.get_metrics(),
    )

  def test_observe_multiple_step(self):
    observer = measurement_metrics.MeasurementObserver()
    observer.observe_first(env=_FAKE_ENV, timestep=_TIMESTEP)
    observer.observe(env=_FAKE_ENV, timestep=_TIMESTEP, action=np.array([1]))
    first_obs_timestep = copy.deepcopy(_TIMESTEP)
    first_obs_timestep.observation = [1000.0, -50.0]
    observer.observe(
        env=_FAKE_ENV, timestep=first_obs_timestep, action=np.array([4]))
    second_obs_timestep = copy.deepcopy(_TIMESTEP)
    second_obs_timestep.observation = [-1000.0, 500.0]
    observer.observe(
        env=_FAKE_ENV, timestep=second_obs_timestep, action=np.array([4]))
    self.assertEqual(
        {
            'measurement[0]_max': 1000.0,
            'measurement[0]_mean': 1.0/3,
            'measurement[0]_p25': -499.5,
            'measurement[0]_p50': 1.0,
            'measurement[0]_p75': 500.5,
            'measurement[1]_max': 500.0,
            'measurement[1]_mean': 448.0/3.0,
            'measurement[1]_p25': -26.0,
            'measurement[1]_p50': -2.0,
            'measurement[1]_p75': 249.0,
            'measurement[0]_min': -1000.0,
            'measurement[1]_min': -50.0,
        },
        observer.get_metrics(),
    )

  def test_observe_empty_observation(self):
    observer = measurement_metrics.MeasurementObserver()
    empty_timestep = copy.deepcopy(_TIMESTEP)
    empty_timestep.observation = {}
    observer.observe_first(env=_FAKE_ENV, timestep=empty_timestep)
    self.assertEqual({}, observer.get_metrics())

  def test_observe_single_dimensions(self):
    observer = measurement_metrics.MeasurementObserver()
    observer.observe_first(env=_FAKE_ENV, timestep=_TIMESTEP)
    single_obs_timestep = copy.deepcopy(_TIMESTEP)
    single_obs_timestep.observation = [1000.0, -50.0]

    observer.observe(
        env=_FAKE_ENV,
        timestep=single_obs_timestep,
        action=np.array([[1, 2], [3, 4]]))

    np.testing.assert_equal(
        {
            'measurement[0]_max': 1000.0,
            'measurement[0]_min': 1000.0,
            'measurement[0]_mean': 1000.0,
            'measurement[0]_p25': 1000.0,
            'measurement[0]_p50': 1000.0,
            'measurement[0]_p75': 1000.0,
            'measurement[1]_max': -50.0,
            'measurement[1]_mean': -50.0,
            'measurement[1]_p25': -50.0,
            'measurement[1]_p50': -50.0,
            'measurement[1]_p75': -50.0,
            'measurement[1]_min': -50.0,
        },
        observer.get_metrics(),
    )


if __name__ == '__main__':
  absltest.main()
