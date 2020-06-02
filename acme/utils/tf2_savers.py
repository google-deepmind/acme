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

"""Utility classes for saving model checkpoints and snapshots."""

import abc
import datetime
import os
import pickle
import signal
import time
from typing import Mapping, Union

from absl import logging
from acme import core
from acme.utils import paths
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree

from tensorflow.python.saved_model import revived_types

PythonState = tf.train.experimental.PythonState
Checkpointable = Union[tf.Module, tf.Variable, PythonState]

_DEFAULT_CHECKPOINT_TTL = int(datetime.timedelta(days=5).total_seconds())
_DEFAULT_SNAPSHOT_TTL = int(datetime.timedelta(days=90).total_seconds())


class TFSaveable(abc.ABC):
  """An interface for objects that expose their checkpointable TF state."""

  @property
  @abc.abstractmethod
  def state(self) -> Mapping[str, Checkpointable]:
    """Returns TensorFlow checkpointable state."""


class Checkpointer:
  """Convenience class for periodically checkpointing.

  This can be used to checkpoint any object with trackable state (e.g.
  tensorflow variables or modules); see tf.train.Checkpoint for
  details. Objects inheriting from tf.train.experimental.PythonState can also
  be checkointed.

  Typically people use Checkpointer to make sure that they can correctly recover
  from a machine going down during learning. For more permanent storage of self-
  contained "networks" see the Snapshotter object.

  Usage example:

  ```python
  model = snt.Linear(10)
  checkpointer = Checkpointer(objects_to_save={'model': model})

  for _ in range(100):
    # ...
    checkpointer.save()
  ```
  """

  def __init__(
      self,
      objects_to_save: Mapping[str, Union[Checkpointable, core.Saveable]],
      *,
      directory: str = '~/acme/',
      subdirectory: str = 'default',
      time_delta_minutes: float = 10.0,
      enable_checkpointing: bool = True,
      add_uid: bool = True,
      max_to_keep: int = 1,
      checkpoint_ttl_seconds: int = _DEFAULT_CHECKPOINT_TTL,
      keep_checkpoint_every_n_hours: int = None,
  ):
    """Builds the saver object.

    Args:
      objects_to_save: Mapping specifying what to checkpoint.
      directory: Which directory to put the checkpoint in.
      subdirectory: Sub-directory to use (e.g. if multiple checkpoints are being
        saved).
      time_delta_minutes: How often to save the checkpoint, in minutes.
      enable_checkpointing: whether to checkpoint or not.
      add_uid: If True adds a UID to the checkpoint path, see
        `paths.get_unique_id()` for how this UID is generated.
      max_to_keep: The maximum number of checkpoints to keep.
      checkpoint_ttl_seconds: TTL (time to leave) in seconds for checkpoints.
      keep_checkpoint_every_n_hours: keep_checkpoint_every_n_hours passed to
        tf.train.CheckpointManager.
    """

    # Convert `Saveable` objects to TF `Checkpointable` first, if necessary.
    def to_ckptable(x: Union[Checkpointable, core.Saveable]) -> Checkpointable:
      if isinstance(x, core.Saveable):
        return SaveableAdapter(x)
      return x

    objects_to_save = {k: to_ckptable(v) for k, v in objects_to_save.items()}

    self._time_delta_minutes = time_delta_minutes
    self._last_saved = 0.
    self._enable_checkpointing = enable_checkpointing
    self._checkpoint_manager = None

    if enable_checkpointing:
      # Checkpoint object that handles saving/restoring.
      self._checkpoint = tf.train.Checkpoint(**objects_to_save)
      self._checkpoint_dir = paths.process_path(
          directory,
          'checkpoints',
          subdirectory,
          ttl_seconds=checkpoint_ttl_seconds,
          backups=False,
          add_uid=add_uid)

      # Create a manager to maintain different checkpoints.
      self._checkpoint_manager = tf.train.CheckpointManager(
          self._checkpoint,
          directory=self._checkpoint_dir,
          max_to_keep=max_to_keep,
          keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

      self.restore()

  def save(self, force: bool = False) -> bool:
    """Save the checkpoint if it's the appropriate time, otherwise no-ops.

    Args:
      force: Whether to force a save regardless of time elapsed since last save.

    Returns:
      A boolean indicating if a save event happened.
    """
    if not self._enable_checkpointing:
      return False

    if (not force and
        time.time() - self._last_saved < 60 * self._time_delta_minutes):
      return False

    # Save any checkpoints.
    logging.info('Saving checkpoint: %s', self._checkpoint_manager.directory)
    self._checkpoint_manager.save()
    self._last_saved = time.time()

    return True

  def restore(self):
    # Restore from the most recent checkpoint (if it exists).
    checkpoint_to_restore = self._checkpoint_manager.latest_checkpoint
    logging.info('Attempting to restoring checkpoint: %s',
                 checkpoint_to_restore)
    self._checkpoint.restore(checkpoint_to_restore)


class CheckpointingRunner(core.Worker):
  """Wrap an object and expose a run method which checkpoints periodically.

  This internally creates a Checkpointer around `wrapped` object and exposes
  all of the methods of `wrapped`. Additionally, anay `**kwargs` passed to the
  runner are forwarded to the internal Checkpointer.
  """

  def __init__(
      self,
      wrapped: Union[Checkpointable, core.Saveable, core.Learner, TFSaveable],
      *,
      time_delta_minutes: int = 30,
      **kwargs,
  ):

    if isinstance(wrapped, TFSaveable):
      # If the object to be wrapped exposes its TF State, checkpoint that.
      objects_to_save = wrapped.state
    else:
      # Otherwise checkpoint the wrapped object itself.
      objects_to_save = wrapped

    self._wrapped = wrapped
    self._time_delta_minutes = time_delta_minutes
    self._checkpointer = Checkpointer(
        objects_to_save={'wrapped': objects_to_save},
        time_delta_minutes=time_delta_minutes,
        **kwargs)

  def run(self):
    """Runs the checkpointer."""

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

    if isinstance(self._wrapped, core.Learner):
      # Learners have a step() method, so alternate between that and ckpt call.
      while True:
        self._wrapped.step()
        self._checkpointer.save()
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


class Snapshotter:
  """Convenience class for periodically snapshotting.

  Objects which can be snapshotted are limited to Sonnet or tensorflow Modules
  which implement a a __call__ method. This will save the module's graph and
  variables such that they can be loaded later using `tf.saved_model.load`. See
  https://www.tensorflow.org/guide/saved_model for more details.

  The Snapshotter is typically used to save infrequent permanent self-contained
  snapshots which can be loaded later for inspection. For frequent saving of
  model parameters in order to guard against pre-emption of the learning process
  see the Checkpointer class.

  Usage example:

  ```python
  model = snt.Linear(10)
  snapshotter = Snapshotter(objects_to_save={'model': model})

  for _ in range(100):
    # ...
    snapshotter.save()
  ```
  """

  def __init__(
      self,
      objects_to_save: Mapping[str, snt.Module],
      *,
      directory: str = '~/acme/',
      time_delta_minutes: float = 30.0,
      snapshot_ttl_seconds: int = _DEFAULT_SNAPSHOT_TTL,
  ):
    """Builds the saver object.

    Args:
      objects_to_save: Mapping specifying what to snapshot.
      directory: Which directory to put the snapshot in.
      time_delta_minutes: How often to save the snapshot, in minutes.
      snapshot_ttl_seconds: TTL (time to leave) in seconds for snapshots.
    """
    objects_to_save = objects_to_save or {}

    self._time_delta_minutes = time_delta_minutes
    self._last_saved = 0.
    self._snapshots = {}

    # Save the base directory path so we can refer to it if needed.
    self.directory = paths.process_path(
        directory, 'snapshots', ttl_seconds=snapshot_ttl_seconds)

    # Save a dictionary mapping paths to snapshot capable models.
    for name, module in objects_to_save.items():
      path = os.path.join(self.directory, name)
      self._snapshots[path] = make_snapshot(module)

  def save(self, force: bool = False) -> bool:
    """Snapshots if it's the appropriate time, otherwise no-ops.

    Args:
      force: If True, save new snapshot no matter how long it's been since the
        last one.

    Returns:
      A boolean indicating if a save event happened.
    """
    seconds_since_last = time.time() - self._last_saved
    if (self._snapshots and
        (force or seconds_since_last >= 60 * self._time_delta_minutes)):
      # Save any snapshots.
      for path, snapshot in self._snapshots.items():
        tf.saved_model.save(snapshot, path)

      # Record the time we finished saving.
      self._last_saved = time.time()

      return True

    return False


class Snapshot(tf.Module):
  """Thin wrapper which allows the module to be saved."""

  def __init__(self):
    super().__init__()
    self._module = None
    self._variables = None
    self._trainable_variables = None

  @tf.function
  def __call__(self, *args, **kwargs):
    return self._module(*args, **kwargs)

  @property
  def submodules(self):
    return [self._module]

  @property
  def variables(self):
    return self._variables

  @property
  def trainable_variables(self):
    return self._trainable_variables


# Registers the Snapshot object above such that when it is restored by
# tf.saved_model.load it will be restored as a Snapshot. This is important
# because it allows us to expose the __call__, and *_variables properties.
revived_types.register_revived_type(
    'acme_snapshot',
    lambda obj: isinstance(obj, Snapshot),
    versions=[
        revived_types.VersionedTypeRegistration(
            object_factory=lambda proto: Snapshot(),
            version=1,
            min_producer_version=1,
            min_consumer_version=1,
            setter=setattr,
        )
    ])


def make_snapshot(module: snt.Module):
  """Create a thin wrapper around a module to make it snapshottable."""
  # Get the input signature as long as it has been created.
  input_signature = _get_input_signature(module)

  # This function will return the object as a composite tensor if it is a
  # distribution and will otherwise return it with no changes.
  def as_composite(obj):
    if isinstance(obj, tfp.distributions.Distribution):
      return tfp.experimental.as_composite(obj)
    else:
      return obj

  # Replace any distributions returned by the module with composite tensors and
  # wrap it up in tf.function so we can process it properly.
  @tf.function
  def wrapped_module(*args, **kwargs):
    return tree.map_structure(as_composite, module(*args, **kwargs))

  # pylint: disable=protected-access
  snapshot = Snapshot()
  snapshot._module = wrapped_module
  snapshot._variables = module.variables
  snapshot._trainable_variables = module.trainable_variables
  # pylint: disable=protected-access

  # Make sure the snapshot has the proper input signature.
  snapshot.__call__.get_concrete_function(*input_signature)

  # If we are an RNN also save the initial-state generating function.
  if isinstance(module, snt.RNNCore):
    snapshot.initial_state = tf.function(module.initial_state)
    snapshot.initial_state.get_concrete_function(
        tf.TensorSpec(shape=(), dtype=tf.int32))

  return snapshot


def _get_input_signature(module: snt.Module):
  """Get module input signature.

  Works even if the module with signature is wrapper into snt.Sequentual or
  snt.DeepRNN.

  Args:
    module: the module which input signature we need to get. The module has to
      either have input_signature itself (i.e. you have to run create_variables
      on the module), or it has to be a module (with input_signature) wrapped in
      (one or multiple) snt.Sequential or snt.DeepRNNs.

  Returns:
    Input signature of the module or None if it's not available.
  """
  if hasattr(module, '_input_signature'):
    return module._input_signature  # pylint: disable=protected-access

  if isinstance(module, snt.Sequential):
    first_layer = module._layers[0]  # pylint: disable=protected-access
    return _get_input_signature(first_layer)

  if isinstance(module, snt.DeepRNN):
    first_layer = module._layers[0]  # pylint: disable=protected-access
    input_signature = _get_input_signature(first_layer)

    # Wrapping a module in DeepRNN changes its state shape, so we need to bring
    # it up to date.
    state = module.initial_state(1)
    input_signature[-1] = tree.map_structure(
        lambda t: tf.TensorSpec((None,) + t.shape[1:], t.dtype), state)

    return input_signature

  # If we get here we can't determine the input signature. So give up.
  raise ValueError('module instance has no input_signature attribute; run '
                   'create_variables to add this annotation.')


class SaveableAdapter(tf.train.experimental.PythonState):
  """Adapter which allows `Saveable` object to be checkpointed by TensorFlow."""

  def __init__(self, object_to_save: core.Saveable):
    self._object_to_save = object_to_save

  def serialize(self):
    state = self._object_to_save.save()
    return pickle.dumps(state)

  def deserialize(self, pickled: bytes):
    state = pickle.loads(pickled)
    self._object_to_save.restore(state)
