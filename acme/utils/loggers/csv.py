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

"""A simple CSV logger."""

import csv
import os
import time

from absl import logging

from acme.utils import paths
from acme.utils.loggers import base


class CSVLogger(base.Logger):
  """Standard CSV logger."""

  _open = open

  def __init__(self,
               directory: str = '~/acme',
               label: str = '',
               time_delta: float = 0.):
    directory = paths.process_path(directory, 'logs', label, add_uid=True)
    self._file_path = os.path.join(directory, 'logs.csv')
    logging.info('Logging to %s', self._file_path)
    self._time = time.time()
    self._time_delta = time_delta
    self._header_exists = False

  def write(self, data: base.LoggingData):
    """Writes a `data` into a row of comma-separated values."""

    # Only log if `time_delta` seconds have passed since last logging event.
    now = time.time()
    if now - self._time < self._time_delta:
      return
    self._time = now

    # Append row to CSV.
    with self._open(self._file_path, mode='a') as f:
      data = base.to_numpy(data)
      keys = sorted(data.keys())
      writer = csv.DictWriter(f, fieldnames=keys)
      if not self._header_exists:
        # Only write the column headers once.
        writer.writeheader()
        self._header_exists = True
      writer.writerow(data)

  @property
  def file_path(self) -> str:
    return self._file_path
