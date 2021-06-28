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

"""Logger for writing to an in-memory list.

This is convenient for e.g. interactive usage via Google Colab.

For example, for usage with pandas:

```python
from acme.utils import loggers
import pandas as pd

logger = InMemoryLogger()
# ...
logger.write({'foo': 1.337, 'bar': 420})

results = pd.DataFrame(logger.data)
```
"""

from typing import Sequence

from acme.utils.loggers import base


class InMemoryLogger(base.Logger):
  """A simple logger that keeps all data in memory."""

  def __init__(self):
    self._data = []

  def write(self, data: base.LoggingData):
    self._data.append(data)

  def close(self):
    pass

  @property
  def data(self) -> Sequence[base.LoggingData]:
    return self._data
