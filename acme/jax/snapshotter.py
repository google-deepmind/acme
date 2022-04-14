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
from typing import Callable, Dict, Sequence, Tuple

from absl import logging
from acme import core
from acme.jax import types
from acme.utils import signals
from acme.utils import paths
from jax.experimental import jax2tf
import tensorflow as tf


class JAXSnapshotter(core.Worker):
  """Periodically fetches new version of params and stores tf.saved_models."""

  # NOTE: External contributor please refrain from modifying the high level of
  # the API defined here.

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
    # To make sure models are captured as close as possible from the same time
    # we gather all the `ModelToSnapshot` in a 1st loop. We then convert/saved
    # them in another loop as this operation can be slow.
    models_and_paths = self._get_models_and_paths(
        saving_time=time.strftime('%Y%m%d-%H%M%S'))

    for model, saving_path in models_and_paths:
      self._snapshot_model(model=model, saving_path=saving_path)

  def _get_models_and_paths(
      self, saving_time: str) -> Sequence[Tuple[types.ModelToSnapshot, str]]:
    """Gets the models to save asssociated with their saving path."""
    models_and_paths = []
    for name, model_fn in self._models.items():
      model = model_fn(self._variable_source)
      model_path = os.path.join(self._path, saving_time, name)
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
