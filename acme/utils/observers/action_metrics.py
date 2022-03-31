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

"""An observer that tracks statistics about the actions."""

from typing import Dict

from acme.utils.observers import base
import dm_env
import numpy as np


class ContinuousActionObserver(base.EnvLoopObserver):
  """Observer that tracks statstics of continuous actions taken by the agent.

  Assumes the action is a np.ndarray, and for each dimension in the action,
  calculates some useful statistics for a particular episode.
  """

  def __init__(self):
    self._actions = None

  def observe_first(self, env: dm_env.Environment,
                    timestep: dm_env.TimeStep) -> None:
    """Observes the initial state."""
    self._actions = []

  def observe(self, env: dm_env.Environment, timestep: dm_env.TimeStep,
              action: np.ndarray) -> None:
    """Records one environment step."""
    self._actions.append(action)

  def get_metrics(self) -> Dict[str, base.Number]:
    """Returns metrics collected for the current episode."""
    aggregate_metrics = {}
    if not self._actions:
      return aggregate_metrics

    metrics = {
        'action_max': np.max(self._actions, axis=0),
        'action_min': np.min(self._actions, axis=0),
        'action_mean': np.mean(self._actions, axis=0),
        'action_p50': np.percentile(self._actions, q=50., axis=0)
    }

    for index, sub_action_metric in np.ndenumerate(metrics['action_max']):
      aggregate_metrics[f'action{list(index)}_max'] = sub_action_metric
      aggregate_metrics[f'action{list(index)}_min'] = metrics['action_min'][
          index]
      aggregate_metrics[f'action{list(index)}_mean'] = metrics['action_mean'][
          index]
      aggregate_metrics[f'action{list(index)}_p50'] = metrics['action_p50'][
          index]

    return aggregate_metrics
