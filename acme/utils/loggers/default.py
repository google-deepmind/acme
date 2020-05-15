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

"""Default logger."""

from acme.utils.loggers import aggregators
from acme.utils.loggers import base
from acme.utils.loggers import csv
from acme.utils.loggers import filters
from acme.utils.loggers import terminal


def make_default_logger(
    label: str,
    save_data: bool = True,
    time_delta: float = 1.0,
) -> base.Logger:
  """Make a default Acme logger.

  Args:
    label: Name to give to the logger.
    save_data: Ignored.
    time_delta: Time (in seconds) between logging events.

  Returns:
    A logger (pipe) object that responds to logger.write(some_dict).
  """
  terminal_logger = terminal.TerminalLogger(label=label, time_delta=time_delta)

  loggers = [terminal_logger]
  if save_data:
    loggers.append(csv.CSVLogger(label))

  logger = aggregators.Dispatcher(loggers)
  logger = filters.NoneFilter(logger)
  logger = filters.TimeFilter(logger, time_delta)
  return logger
