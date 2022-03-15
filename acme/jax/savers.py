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
import time
from typing import Any, Callable, Dict

from absl import logging
from acme import core
from acme.jax import types
from acme.tf import savers as tf_savers
from acme.utils import signals
from acme.utils import paths
from jax.experimental import jax2tf
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
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


def model_to_tf_module(model: types.ModelToSnapshot) -> tf.Module:
  def jax_fn_to_save(**kwargs):
    return model.model(model.params, **kwargs)

  module = tf.Module()
  module.f = tf.function(jax2tf.convert(jax_fn_to_save), autograph=False)
  # Traces input to ensure the model has the correct shapes.
  module.f(**model.dummy_kwargs)
  return module


class JAX2TFSaver(core.Worker):
  """Periodically fetches new version of params and stores tf.saved_models."""

  def __init__(self,
               variable_source: core.VariableSource,
               models: Dict[str, Callable[[core.VariableSource],
                                          types.ModelToSnapshot]],
               path: str,
               add_uid: bool = False):
    self._variable_source = variable_source
    self._models = models
    self._path = paths.process_path(path, add_uid=add_uid)

  # Handle preemption signal. Note that this must happen in the main thread.
  def _signal_handler(self):
    logging.info('Caught SIGTERM: forcing models save.')
    self._save()

  def _save(self):
    for name, model_fn in self._models.items():
      model = model_fn(self._variable_source)
      module = model_to_tf_module(model)
      model_path = os.path.join(self._path, time.strftime('%Y%m%d-%H%M%S'),
                                name)
      tf.saved_model.save(module, model_path)

  def run(self):
    """Runs the saver."""
    with signals.runtime_terminator(self._signal_handler):
      while True:
        self._save()
        time.sleep(5 * 60)
