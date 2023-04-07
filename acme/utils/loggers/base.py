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

"""Base logger."""

import abc
from typing import Any, Mapping, Optional

import numpy as np
import tree
from typing_extensions import Protocol

LoggingData = Mapping[str, Any]


class Logger(abc.ABC):
    """A logger has a `write` method."""

    @abc.abstractmethod
    def write(self, data: LoggingData) -> None:
        """Writes `data` to destination (file, terminal, database, etc)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Closes the logger, not expecting any further write."""


TaskInstance = int
# TODO(stanczyk): Turn LoggerLabel into an enum of [Learner, Actor, Evaluator].
LoggerLabel = str
LoggerStepsKey = str


class LoggerFactory(Protocol):
    def __call__(
        self,
        label: LoggerLabel,
        steps_key: Optional[LoggerStepsKey] = None,
        instance: Optional[TaskInstance] = None,
    ) -> Logger:
        ...


class NoOpLogger(Logger):
    """Simple Logger which does nothing and outputs no logs.

  This should be used sparingly, but it can prove useful if we want to quiet an
  individual component and have it produce no logging whatsoever.
  """

    def write(self, data: LoggingData):
        pass

    def close(self):
        pass


def tensor_to_numpy(value: Any):
    if hasattr(value, "numpy"):
        return value.numpy()  # tf.Tensor (TF2).
    if hasattr(value, "device_buffer"):
        return np.asarray(value)  # jnp.DeviceArray.
    return value


def to_numpy(values: Any):
    """Converts tensors in a nested structure to numpy.

  Converts tensors from TensorFlow to Numpy if needed without importing TF
  dependency.

  Args:
    values: nested structure with numpy and / or TF tensors.

  Returns:
    Same nested structure as values, but with numpy tensors.
  """
    return tree.map_structure(tensor_to_numpy, values)
