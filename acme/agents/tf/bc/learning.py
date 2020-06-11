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

"""BC Learner implementation."""

from typing import Dict, List

import acme
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
import numpy as np
import sonnet as snt
import tensorflow as tf


class BCLearner(acme.Learner, tf2_savers.TFSaveable):
  """BC learner.

  This is the learning component of a BC agent. IE it takes a dataset as input
  and implements update functionality to learn from this dataset. Optionally
  it takes a replay client as well to allow for updating of priorities.
  """

  def __init__(self,
               network: snt.Module,
               learning_rate: float,
               dataset: tf.data.Dataset,
               counter: counting.Counter = None,
               logger: loggers.Logger = None):
    """Initializes the learner.

    Args:
      network: the online Q network (the one being optimized)
      learning_rate: learning rate for the q-network update.
      dataset: dataset to learn from.
      counter: Counter object for (potentially distributed) counting.
      logger: Logger object for writing logs to.
    """

    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.TerminalLogger('learner', time_delta=1.)

    # Get an iterator over the dataset.
    self._iterator = iter(dataset)  # pytype: disable=wrong-arg-types
    # TODO(b/155086959): Fix type stubs and remove.

    self._network = network
    self._optimizer = snt.optimizers.Adam(learning_rate)

    self._variables: List[List[tf.Tensor]] = [network.trainable_variables]
    self._num_steps = tf.Variable(0, dtype=tf.int32)

    self._snapshotter = tf2_savers.Snapshotter(
        objects_to_save={'network': network}, time_delta_minutes=60.)

  @tf.function
  def _step(self) -> Dict[str, tf.Tensor]:
    """Do a step of SGD and update the priorities."""

    # Pull out the data needed for updates/priorities.
    inputs = next(self._iterator)
    o_tm1, a_tm1, r_t, d_t, o_t = inputs.data
    del r_t, d_t, o_t

    with tf.GradientTape() as tape:
      # Evaluate our networks.
      logits = self._network(o_tm1)
      cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
      loss = cce(a_tm1, logits)

    gradients = tape.gradient(loss, self._network.trainable_variables)
    self._optimizer.apply(gradients, self._network.trainable_variables)

    self._num_steps.assign_add(1)

    # Compute the global norm of the gradients for logging.
    global_gradient_norm = tf.linalg.global_norm(gradients)
    fetches = {'loss': loss, 'gradient_norm': global_gradient_norm}

    return fetches

  def step(self):
    # Do a batch of SGD.
    result = self._step()

    # Update our counts and record it.
    counts = self._counter.increment(steps=1)
    result.update(counts)

    # Snapshot and attempt to write logs.
    self._snapshotter.save()
    self._logger.write(result)

  def get_variables(self, names: List[str]) -> List[np.ndarray]:
    return tf2_utils.to_numpy(self._variables)

  @property
  def state(self):
    """Returns the stateful parts of the learner for checkpointing."""
    return {
        'network': self._network,
        'optimizer': self._optimizer,
        'num_steps': self._num_steps
    }
