# Lint as: python3
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

"""Utility classes for saving model checkpoints."""

import datetime
import os
import pickle
from typing import Any

from absl import logging
from acme import core
from acme.tf import savers as tf_savers
import jax.numpy as jnp
import numpy as np
import tree

# Internal imports.

CheckpointState = Any

_DEFAULT_CHECKPOINT_TTL = int(datetime.timedelta(days=5).total_seconds())
_ARRAY_NAME = 'array_nest'
_EXEMPLAR_NAME = 'nest_exemplar'


def restore_from_path(ckpt_dir: str) -> CheckpointState:
  """Restore the state stored in ckpt_dir."""
  array_path = os.path.join(ckpt_dir, _ARRAY_NAME)
  exemplar_path = os.path.join(ckpt_dir, _EXEMPLAR_NAME)

  with open(exemplar_path, 'rb') as f:
    exemplar = pickle.load(f)

  with open(array_path, 'rb') as f:
    files = np.load(f, allow_pickle=True)
    flat_state = [files[key] for key in files.files]
  unflattened_tree = tree.unflatten_as(exemplar, flat_state)

  def maybe_convert_to_python(value, numpy):
    return value if numpy else value.item()

  return tree.map_structure(maybe_convert_to_python, unflattened_tree, exemplar)


def save_to_path(ckpt_dir: str, state: CheckpointState):
  """Save the state in ckpt_dir."""

  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

  is_numpy = lambda x: isinstance(x, (np.ndarray, jnp.DeviceArray))
  flat_state = tree.flatten(state)
  nest_exemplar = tree.map_structure(is_numpy, state)

  array_path = os.path.join(ckpt_dir, _ARRAY_NAME)
  logging.info('Saving flattened array nest to %s', array_path)
  def _disabled_seek(*_):
    raise AttributeError('seek() is disabled on this object.')
  with open(array_path, 'wb') as f:
    setattr(f, 'seek', _disabled_seek)
    np.savez(f, *flat_state)

  exemplar_path = os.path.join(ckpt_dir, _EXEMPLAR_NAME)
  logging.info('Saving nest exemplar to %s', exemplar_path)
  with open(exemplar_path, 'wb') as f:
    pickle.dump(nest_exemplar, f)


# Use TF checkpointer.
class Checkpointer(tf_savers.Checkpointer):

  def __init__(
      self,
      object_to_save: core.Saveable,
      directory: str = '~/acme',
      subdirectory: str = 'default',
      **tf_checkpointer_kwargs):
    super().__init__(dict(saveable=object_to_save),
                     directory=directory,
                     subdirectory=subdirectory,
                     **tf_checkpointer_kwargs)


CheckpointingRunner = tf_savers.CheckpointingRunner
