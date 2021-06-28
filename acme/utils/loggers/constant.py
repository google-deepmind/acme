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

"""Logger for values that remain constant."""

from acme.utils.loggers import base


class ConstantLogger(base.Logger):
  """Logger for values that remain constant throughout the experiment.

  This logger is used to log additional values e.g. level_name or
  hyperparameters that do not change in an experiment. Having these values
  allows to group or facet plots when analysing data post-experiment.
  """

  def __init__(
      self,
      constant_data: base.LoggingData,
      to: base.Logger,
  ):
    """Initialise the extra info logger.

    Args:
      constant_data: Key-value pairs containing the constant info to be logged.
      to: The logger to add these extra info to.
    """
    self._constant_data = constant_data
    self._to = to

  def write(self, data: base.LoggingData):
    self._to.write({**self._constant_data, **data})

  def close(self):
    self._to.close()
