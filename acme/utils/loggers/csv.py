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

"""A simple CSV logger.

Warning: Does not support preemption.
"""

import csv
import os
import time
from typing import TextIO, Tuple, Union

from absl import logging

from acme.utils import paths
from acme.utils.loggers import base


class CSVLogger(base.Logger):
  """Standard CSV logger.

  The fields are inferred from the first call to write() and any additional
  fields afterwards are ignored.
  """

  _open = open

  def __init__(self,
               directory_or_file: Union[str, TextIO] = '~/acme',
               label: str = '',
               time_delta: float = 0.,
               add_uid: bool = True):
    self._last_log_time = time.time() - time_delta
    self._time_delta = time_delta
    self._add_uid = add_uid
    self._writer = None
    self._file, self._file_owner = self._create_file(directory_or_file, label)
    logging.info('Logging to %s', self.file_path)

  def _create_file(self, directory_or_file: Union[str, TextIO],
                   label: str) -> Tuple[TextIO, bool]:
    """Opens a file if input is a directory or use existing file."""
    if isinstance(directory_or_file, str):
      directory = paths.process_path(
          directory_or_file, 'logs', label, add_uid=self._add_uid)
      file_path = os.path.join(directory, 'logs.csv')
      file = self._open(file_path, mode='a')
      file_owner = True
    else:
      file = directory_or_file
      file_owner = False
      if label:
        logging.info('File, not directory, passed to CSVLogger; label not '
                     'used.')
      if not file.mode.startswith('a'):
        raise ValueError('File must be open in append mode; instead got '
                         f'{file.mode}.')
    return file, file_owner

  def write(self, data: base.LoggingData):
    """Writes a `data` into a row of comma-separated values."""
    # Only log if `time_delta` seconds have passed since last logging event.
    now = time.time()
    if now - self._last_log_time < self._time_delta:
      return
    self._last_log_time = now

    # Append row to CSV.
    data = base.to_numpy(data)
    # Use fields from initial `data` to create the header. If extra fields are
    # present in subsequent `data`, we ignore them.
    if not self._writer:
      fields = sorted(data.keys())
      self._writer = csv.DictWriter(self._file, fieldnames=fields,
                                    extrasaction='ignore')
      self._writer.writeheader()
    self._writer.writerow(data)

  def close(self):
    if self._file_owner:
      self._file.close()

  @property
  def file_path(self) -> str:
    return self._file.name
