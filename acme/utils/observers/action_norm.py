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

"""An observer that collects action norm stats.
"""
from typing import Dict

from acme.utils.observers import base
import dm_env
import numpy as np


class ActionNormObserver(base.EnvLoopObserver):
  """An observer that collects action norm stats."""

  def __init__(self):
    self._action_norms = None

  def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep
                    ) -> None:
    """Observes the initial state."""
    self._action_norms = []

  def observe(self, env: dm_env.Environment, timestep: dm_env.TimeStep,
              action: np.ndarray) -> None:
    """Records one environment step."""
    self._action_norms.append(np.linalg.norm(action))

  def get_metrics(self) -> Dict[str, base.Number]:
    """Returns metrics collected for the current episode."""
    return {'action_norm_avg': np.mean(self._action_norms),
            'action_norm_min': np.min(self._action_norms),
            'action_norm_max': np.max(self._action_norms)}
