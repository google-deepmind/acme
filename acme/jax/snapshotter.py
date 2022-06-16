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

"""Utility classes for snapshotting models."""

import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from absl import logging
from acme import core
from acme.jax import types
from acme.utils import signals
from acme.utils import paths
from jax.experimental import jax2tf
import tensorflow as tf

# Internal imports.


class JAXSnapshotter(core.Worker):
  """Periodically fetches new version of params and stores tf.saved_models."""

  # NOTE: External contributor please refrain from modifying the high level of
  # the API defined here.

  def __init__(self,
               variable_source: core.VariableSource,
               models: Dict[str, Callable[[core.VariableSource],
                                          types.ModelToSnapshot]],
               path: str,
               max_to_keep: Optional[int] = None,
               add_uid: bool = False):
    self._variable_source = variable_source
    self._models = models
    self._path = paths.process_path(path, add_uid=add_uid)
    self._max_to_keep = max_to_keep
    self._snapshot_paths: Optional[List[str]] = None

  # Handle preemption signal. Note that this must happen in the main thread.
  def _signal_handler(self):
    logging.info('Caught SIGTERM: forcing models save.')
    self._save()

  def _save(self):
    if not self._snapshot_paths:
      # Lazy discovery of already existing snapshots.
      self._snapshot_paths = os.listdir(self._path)
      self._snapshot_paths.sort(key=int, reverse=True)

    snapshot_location = os.path.join(self._path, time.strftime('%Y%m%d-%H%M%S'))
    if self._snapshot_paths and self._snapshot_paths[0] == snapshot_location:
      logging.info('Snapshot for the current time already exists.')
      return

    # To make sure models are captured as close as possible from the same time
    # we gather all the `ModelToSnapshot` in a 1st loop. We then convert/saved
    # them in another loop as this operation can be slow.
    models_and_paths = self._get_models_and_paths(path=snapshot_location)
    self._snapshot_paths.insert(0, snapshot_location)

    for model, saving_path in models_and_paths:
      self._snapshot_model(model=model, saving_path=saving_path)

    # Delete any excess snapshots.
    while self._max_to_keep and len(self._snapshot_paths) > self._max_to_keep:
      paths.rmdir(self._snapshot_paths.pop())

  def _get_models_and_paths(
      self, path: str) -> Sequence[Tuple[types.ModelToSnapshot, str]]:
    """Gets the models to save asssociated with their saving path."""
    models_and_paths = []
    for name, model_fn in self._models.items():
      model = model_fn(self._variable_source)
      model_path = os.path.join(path, name)
      models_and_paths.append((model, model_path))
    return models_and_paths

  def _snapshot_model(
      self, model: types.ModelToSnapshot,
      saving_path: str) -> None:
    module = model_to_tf_module(model)
    tf.saved_model.save(module, saving_path)

  def run(self):
    """Runs the saver."""
    with signals.runtime_terminator(self._signal_handler):
      while True:
        self._save()
        time.sleep(5 * 60)


def model_to_tf_module(model: types.ModelToSnapshot) -> tf.Module:

  def jax_fn_to_save(**kwargs):
    return model.model(model.params, **kwargs)

  module = tf.Module()
  module.f = tf.function(jax2tf.convert(jax_fn_to_save), autograph=False)
  # Traces input to ensure the model has the correct shapes.
  module.f(**model.dummy_kwargs)
  return module
