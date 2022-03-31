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

"""An observer that tracks statistics about the observations."""

from typing import Mapping, List

from acme.utils.observers import base
import dm_env
import numpy as np


class MeasurementObserver(base.EnvLoopObserver):
  """Observer the provides statistics for measurements at every timestep.

  This assumes the measurements is a multidimensional array with a static spec.
  Warning! It is not intended to be used for high dimensional observations.

  self._measurements: List[np.ndarray]
  """

  def __init__(self):
    self._measurements = []

  def observe_first(self, env: dm_env.Environment,
                    timestep: dm_env.TimeStep) -> None:
    """Observes the initial state."""
    self._measurements = []

  def observe(self, env: dm_env.Environment, timestep: dm_env.TimeStep,
              action: np.ndarray) -> None:
    """Records one environment step."""
    self._measurements.append(timestep.observation)

  def get_metrics(self) -> Mapping[str, List[base.Number]]:
    """Returns metrics collected for the current episode."""
    aggregate_metrics = {}
    if not self._measurements:
      return aggregate_metrics

    metrics = {
        'measurement_max': np.max(self._measurements, axis=0),
        'measurement_min': np.min(self._measurements, axis=0),
        'measurement_mean': np.mean(self._measurements, axis=0),
        'measurement_p25': np.percentile(self._measurements, q=25., axis=0),
        'measurement_p50': np.percentile(self._measurements, q=50., axis=0),
        'measurement_p75': np.percentile(self._measurements, q=75., axis=0),
    }
    for index, sub_observation_metric in np.ndenumerate(
        metrics['measurement_max']):
      aggregate_metrics[
          f'measurement{list(index)}_max'] = sub_observation_metric
      aggregate_metrics[f'measurement{list(index)}_min'] = metrics[
          'measurement_min'][index]
      aggregate_metrics[f'measurement{list(index)}_mean'] = metrics[
          'measurement_mean'][index]
      aggregate_metrics[f'measurement{list(index)}_p50'] = metrics[
          'measurement_p50'][index]
      aggregate_metrics[f'measurement{list(index)}_p25'] = metrics[
          'measurement_p25'][index]
      aggregate_metrics[f'measurement{list(index)}_p75'] = metrics[
          'measurement_p75'][index]
    return aggregate_metrics
