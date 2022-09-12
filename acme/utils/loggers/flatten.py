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

"""Logger that flattens nested data."""

from typing import Sequence

from acme.utils.loggers import base


class FlattenDictLogger(base.Logger):
  """Logger which flattens sub-dictionaries into the top level dict."""

  def __init__(self,
               logger: base.Logger,
               label: str = 'Logs',
               raw_keys: Sequence[str] = ()):
    """Initializer.

    Args:
      logger: The wrapped logger.
      label: The label to add as a prefix to all keys except for raw ones.
      raw_keys: The keys that should not be prefixed. The values for these keys
        must always be flat. Metric visualisation tools may require certain
        keys to be present in the logs (e.g. 'step', 'timestamp'), so these
        keys should not be prefixed.
    """
    self._logger = logger
    self._label = label
    self._raw_keys = raw_keys

  def write(self, values: base.LoggingData):
    flattened_values = {}
    for key, value in values.items():
      if key in self._raw_keys:
        flattened_values[key] = value
        continue
      name = f'{self._label}/{key}'
      if isinstance(value, dict):
        for sub_key, sub_value in value.items():
          flattened_values[f'{name}/{sub_key}'] = sub_value
      else:
        flattened_values[name] = value

    self._logger.write(flattened_values)

  def close(self):
    self._logger.close()
