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

"""Utilities for logging to the terminal."""

import time

from acme.utils.loggers import base
import tensorflow as tf


def _format_key(key: str) -> str:
  """Internal function for formatting keys in Tensorboard format."""
  return key.title().replace('_', '')


class TFSummaryLogger(base.Logger):
  """Logs to a tf.summary created in a given logdir.

  If multiple TFSummaryLogger are created with the same logdir, results will be
  categorized by labels.
  """

  def __init__(
      self,
      logdir: str,
      label: str = 'Logs',
  ):
    """Initializes the logger.

    Args:
      logdir: directory to which we should log files.
      label: label string to use when logging. Default to 'Logs'.
    """
    self._time = time.time()
    self.label = label
    self._iter = 0
    self.summary = tf.summary.create_file_writer(logdir)

  def write(self, values: base.LoggingData):
    with self.summary.as_default():
      for key, value in values.items():
        tf.summary.scalar(
            f'{self.label}/{_format_key(key)}',
            value,
            step=self._iter)
    self._iter += 1
