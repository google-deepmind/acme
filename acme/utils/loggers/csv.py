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
from typing import TextIO, Union

from absl import logging

from acme.utils import paths
from acme.utils.loggers import base


class CSVLogger(base.Logger):
  """Standard CSV logger.

  The fields are inferred from the first call to write() and any additional
  fields afterwards are ignored.

  TODO(jaslanides): Consider making this stateless/robust to preemption.
  """

  _open = open

  def __init__(
      self,
      directory_or_file: Union[str, TextIO] = '~/acme',
      label: str = '',
      time_delta: float = 0.,
      add_uid: bool = True,
      flush_every: int = 30,
  ):
    """Instantiates the logger.

    Args:
      directory_or_file: Either a directory path as a string, or a file TextIO
        object.
      label: Extra label to add to logger. This is added as a suffix to the
        directory.
      time_delta: Interval in seconds between which writes are dropped to
        throttle throughput.
      add_uid: Whether to add a UID to the file path. See `paths.process_path`
        for details.
      flush_every: Interval (in writes) between flushes.
    """

    if flush_every <= 0:
      raise ValueError(
          f'`flush_every` must be a positive integer (got {flush_every}).')

    self._last_log_time = time.time() - time_delta
    self._time_delta = time_delta
    self._flush_every = flush_every
    self._add_uid = add_uid
    self._writer = None
    self._file_owner = False
    self._file = self._create_file(directory_or_file, label)
    self._writes = 0
    logging.info('Logging to %s', self.file_path)

  def _create_file(
      self,
      directory_or_file: Union[str, TextIO],
      label: str,
  ) -> TextIO:
    """Opens a file if input is a directory or use existing file."""
    if isinstance(directory_or_file, str):
      directory = paths.process_path(
          directory_or_file, 'logs', label, add_uid=self._add_uid)
      file_path = os.path.join(directory, 'logs.csv')
      self._file_owner = True
      return self._open(file_path, mode='a')

    # TextIO instance.
    file = directory_or_file
    if label:
      logging.info('File, not directory, passed to CSVLogger; label not used.')
    if not file.mode.startswith('a'):
      raise ValueError('File must be open in append mode; instead got '
                       f'mode="{file.mode}".')
    return file

  def write(self, data: base.LoggingData):
    """Writes a `data` into a row of comma-separated values."""
    # Only log if `time_delta` seconds have passed since last logging event.
    now = time.time()

    # TODO(b/192227744): Remove this in favour of filters.TimeFilter.
    elapsed = now - self._last_log_time
    if elapsed < self._time_delta:
      logging.debug('Not due to log for another %.2f seconds, dropping data.',
                    self._time_delta - elapsed)
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
      # Write header only if the file is empty.
      if not self._file.tell():
        self._writer.writeheader()
    self._writer.writerow(data)

    # Flush every `flush_every` writes.
    if self._writes % self._flush_every == 0:
      self.flush()
    self._writes += 1

  def close(self):
    self.flush()
    if self._file_owner:
      self._file.close()

  def flush(self):
    self._file.flush()

  @property
  def file_path(self) -> str:
    return self._file.name
