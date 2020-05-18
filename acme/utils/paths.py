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

"""Filesystem path helpers."""

import os
import os.path
import sys
from typing import Optional, Tuple
import uuid

from absl import flags

flags.DEFINE_string('acme_id', None, 'Experiment identifier to use for Acme.')
FLAGS = flags.FLAGS

# Pre-compute a unique identifier which is consistent within a single process.
_ACME_ID = uuid.uuid1()


def process_path(path: str,
                 *subpaths: str,
                 ttl_seconds: Optional[int] = None,
                 backups: Optional[bool] = None,
                 add_uid: bool = True) -> str:
  """Process the path string.

  This will process the path string by running `os.path.expanduser` to replace
  any initial "~". It will also append a unique string on the end of the path
  and create the directories leading to this path if necessary.

  Args:
    path: string defining the path to process and create.
    *subpaths: potential subpaths to include after uniqification.
    ttl_seconds: ignored.
    backups: ignored.
    add_uid: Whether to add a unique directory identifier between `path` and
      `subpaths`. If FLAGS.acme_id is set, will use that as the identifier.

  Returns:
    the processed, expanded path string.
  """
  del backups, ttl_seconds

  path = os.path.expanduser(path)
  # TODO(b/145460917): consider replacing this---e.g. with a timestamp.
  if add_uid:
    path = os.path.join(path, *get_unique_id())
  path = os.path.join(path, *subpaths)
  os.makedirs(path, exist_ok=True)
  return path


def get_unique_id() -> Tuple[str, ...]:
  """Makes a unique identifier for this process; override with FLAGS.acme_id."""
  try:
    FLAGS.acme_id
  except flags.UnparsedFlagAccessError:
    # Parse flags if they haven't been parsed already (e.g. if under pytest).
    FLAGS(sys.argv)
  identifier = FLAGS.acme_id or str(_ACME_ID)
  return (identifier,)
