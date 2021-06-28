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

"""An image logger, for writing out arrays to disk as PNG."""

import collections
import pathlib
from typing import Optional

from absl import logging
from acme.utils.loggers import base
from PIL import Image


class ImageLogger(base.Logger):
  """Logger for writing NumPy arrays as PNG images to disk.

  Assumes that all data passed are NumPy arrays that can be converted to images.

  TODO(jaslanides): Make this stateless/robust to preemptions.
  """

  def __init__(
      self,
      directory: str,
      *,
      label: str = '',
      mode: Optional[str] = None,
  ):
    """Initialises the writer.

    Args:
      directory: Base directory to which images are logged.
      label: Optional subdirectory in which to save images.
      mode: Image mode for use with Pillow. If `None` (default), mode is
        determined by data type. See [0] for details.

    [0] https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes
    """

    self._path = self._get_path(directory, label)
    if not self._path.exists():
      self._path.mkdir(parents=True)

    self._mode = mode
    self._indices = collections.defaultdict(int)

  def write(self, data: base.LoggingData):
    for k, v in data.items():
      image = Image.fromarray(v, mode=self._mode)
      path = self._path / f'{k}_{self._indices[k]:06}.png'
      self._indices[k] += 1
      with path.open(mode='wb') as f:
        logging.info('Writing image to %s.', str(path))
        image.save(f)

  def close(self):
    pass

  @property
  def directory(self) -> str:
    return str(self._path)

  def _get_path(self, *args, **kwargs) -> pathlib.Path:
    return pathlib.Path(*args, **kwargs)
