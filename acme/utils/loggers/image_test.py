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

"""Tests for image logger."""

import os

from acme.testing import test_utils
from acme.utils.loggers import image
import numpy as np
from PIL import Image

from absl.testing import absltest


class ImageTest(test_utils.TestCase):

  def test_save_load_identity(self):
    directory = self.get_tempdir()
    logger = image.ImageLogger(directory, label='foo')
    array = (np.random.rand(10, 10) * 255).astype(np.uint8)
    logger.write({'img': array})

    with open(f'{directory}/foo/img_000000.png', mode='rb') as f:
      out = np.asarray(Image.open(f))
    np.testing.assert_array_equal(array, out)

  def test_indexing(self):
    directory = self.get_tempdir()
    logger = image.ImageLogger(directory, label='foo')
    zeros = np.zeros(shape=(3, 3), dtype=np.uint8)
    logger.write({'img': zeros, 'other_img': zeros + 1})
    logger.write({'img': zeros - 1})
    logger.write({'other_img': zeros + 1})
    logger.write({'other_img': zeros + 2})

    fnames = sorted(os.listdir(f'{directory}/foo'))
    expected = [
        'img_000000.png',
        'img_000001.png',
        'other_img_000000.png',
        'other_img_000001.png',
        'other_img_000002.png',
    ]
    self.assertEqual(fnames, expected)


if __name__ == '__main__':
  absltest.main()
