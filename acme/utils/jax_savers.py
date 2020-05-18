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
import signal
import threading
import time
from typing import Any, TypeVar, Union

from absl import logging
from acme import core
from acme.utils import paths
import jax.numpy as jnp
import numpy as np
import tree

# Internal imports.

Number = Union[int, float]
CheckpointState = Any
T = TypeVar('T')

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
    return value if numpy else np.asscalar(value)

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


class Checkpointer:
  """Convenience class for periodically checkpointing.

  This can be used to checkpoint any numpy arrays or any object which is
  pickelable.
  """

  def __init__(
      self,
      object_to_save: core.Saveable,
      directory: str = '~/acme/',
      subdirectory: str = 'default',
      time_delta_minutes: float = 10.,
      add_uid: bool = True,
      checkpoint_ttl_seconds: int = _DEFAULT_CHECKPOINT_TTL,
  ):
    """Builds the saver object.

    Args:
      object_to_save: The object to save in this checkpoint, this must have a
        save and restore method.
      directory: Which directory to put the checkpoint in.
      subdirectory: Sub-directory to use (e.g. if multiple checkpoints are being
        saved).
      time_delta_minutes: How often to save the checkpoint, in minutes.
      add_uid: If True adds a UID to the checkpoint path, see
        `paths.get_unique_id()` for how this UID is generated.
      checkpoint_ttl_seconds: TTL (time to live) in seconds for checkpoints.
    """
    # TODO(tamaranorman) accept a Union[Saveable, Mapping[str, Saveable]] here
    self._object_to_save = object_to_save
    self._time_delta_minutes = time_delta_minutes

    self._last_saved = 0.
    self._lock = threading.Lock()

    self._checkpoint_dir = paths.process_path(
        directory,
        'checkpoints',
        subdirectory,
        ttl_seconds=checkpoint_ttl_seconds,
        backups=False,
        add_uid=add_uid)

    # Restore from the most recent checkpoint (if it exists).
    self.restore()

  def restore(self):
    """Restores from the saved checkpoint if it exists."""
    if os.path.exists(os.path.join(self._checkpoint_dir, _EXEMPLAR_NAME)):
      logging.info('Restoring checkpoint: %s', self._checkpoint_dir)
      with self._lock:
        state = restore_from_path(self._checkpoint_dir)
        self._object_to_save.restore(state)

  def save(self, force: bool = False) -> bool:
    """Save the checkpoint if it's the appropriate time, otherwise no-ops.

    Args:
      force: Whether to force a save regardless of time elapsed since last save.

    Returns:
      A boolean indicating if a save event happened.
    """

    if (not force and
        time.time() - self._last_saved < 60 * self._time_delta_minutes):
      return False

    logging.info('Saving checkpoint: %s', self._checkpoint_dir)
    with self._lock:
      state = self._object_to_save.save()
      save_to_path(self._checkpoint_dir, state)

    self._last_saved = time.time()
    return True


class CheckpointingRunner(core.Worker):
  """Wrap an object and checkpoints periodically.

  This is either uses the run method if one doesn't exist or performs it in a
  thread.

  This internally creates a Checkpointer around `wrapped` object and exposes
  all of the methods of `wrapped`. Additionally, any `**kwargs` passed to the
  runner are forwarded to the internal Checkpointer.
  """

  def __init__(
      self,
      wrapped: Union[core.Saveable, core.Worker],
      *,
      time_delta_minutes: float = 10.,
      **kwargs,
  ):
    self._wrapped = wrapped
    self._time_delta_minutes = time_delta_minutes
    self._checkpointer = Checkpointer(
        object_to_save=wrapped, time_delta_minutes=1, **kwargs)

  def run(self):
    """Periodically checkpoints the given object."""

    # Handle preemption signal. Note that this must happen in the main thread.
    def _signal_handler(signum: signal.Signals, frame):
      del signum, frame
      logging.info('Caught SIGTERM: forcing a checkpoint save.')
      self._checkpointer.save(force=True)

    try:
      signal.signal(signal.SIGTERM, _signal_handler)
    except ValueError:
      logging.warning(
          'Caught ValueError when registering signal handler. '
          'This probably means we are not running in the main thread. '
          'Proceeding without checkpointing-on-preemption.')

    if isinstance(self._wrapped, core.Worker):
      # Do checkpointing in a separate thread and defer to worker's run().
      threading.Thread(target=self.checkpoint).start()
      self._wrapped.run()
    else:
      # Wrapped object doesn't have a run method; set our run method to ckpt.
      self.checkpoint()

  def __dir__(self):
    return dir(self._wrapped)

  def __getattr__(self, name):
    return getattr(self._wrapped, name)

  def checkpoint(self):
    while True:
      self._checkpointer.save()
      time.sleep(self._time_delta_minutes * 60)
