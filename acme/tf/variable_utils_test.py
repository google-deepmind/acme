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

"""Tests for acme.tf.variable_utils."""

import threading

from acme.testing import fakes
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
import sonnet as snt
import tensorflow as tf


_MLP_LAYERS = [50, 30]
_INPUT_SIZE = 28
_BATCH_SIZE = 8
_UPDATE_PERIOD = 2


class VariableClientTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    # Create two instances of the same model.
    self._actor_model = snt.nets.MLP(_MLP_LAYERS)
    self._learner_model = snt.nets.MLP(_MLP_LAYERS)

    # Create variables first.
    input_spec = tf.TensorSpec(shape=(_INPUT_SIZE,), dtype=tf.float32)
    tf2_utils.create_variables(self._actor_model, [input_spec])
    tf2_utils.create_variables(self._learner_model, [input_spec])

  def test_update_and_wait(self):
    # Create a variable source (emulating the learner).
    np_learner_variables = tf2_utils.to_numpy(self._learner_model.variables)
    variable_source = fakes.VariableSource(np_learner_variables)

    # Create a variable client (emulating the actor).
    variable_client = tf2_variable_utils.VariableClient(
        variable_source, {'policy': self._actor_model.variables})

    # Create some random batch of test input:
    x = tf.random.normal(shape=(_BATCH_SIZE, _INPUT_SIZE))

    # Before copying variables, the models have different outputs.
    self.assertNotAllClose(self._actor_model(x), self._learner_model(x))

    # Update the variable client.
    variable_client.update_and_wait()

    # After copying variables (by updating the client), the models are the same.
    self.assertAllClose(self._actor_model(x), self._learner_model(x))

  def test_update(self):
    # Create a barrier to be shared between the test body and the variable
    # source. The barrier will block until, in this case, two threads call
    # wait(). Note that the (fake) variable source will call it within its
    # get_variables() call.
    barrier = threading.Barrier(2)

    # Create a variable source (emulating the learner).
    np_learner_variables = tf2_utils.to_numpy(self._learner_model.variables)
    variable_source = fakes.VariableSource(np_learner_variables, barrier)

    # Create a variable client (emulating the actor).
    variable_client = tf2_variable_utils.VariableClient(
        variable_source, {'policy': self._actor_model.variables},
        update_period=_UPDATE_PERIOD)

    # Create some random batch of test input:
    x = tf.random.normal(shape=(_BATCH_SIZE, _INPUT_SIZE))

    # Create variables by doing the computation once.
    learner_output = self._learner_model(x)
    actor_output = self._actor_model(x)
    del learner_output, actor_output

    for _ in range(_UPDATE_PERIOD):
      # Before the update period is reached, the models have different outputs.
      self.assertNotAllClose(self._actor_model.variables,
                             self._learner_model.variables)

      # Before the update period is reached, the variable client should not make
      # any requests for variables.
      self.assertIsNone(variable_client._future)

      variable_client.update()

    # Make sure the last call created a request for variables and reset the
    # internal call counter.
    self.assertIsNotNone(variable_client._future)
    self.assertEqual(variable_client._call_counter, 0)
    future = variable_client._future

    for _ in range(_UPDATE_PERIOD):
      # Before the barrier allows the variables to be released, the models have
      # different outputs.
      self.assertNotAllClose(self._actor_model.variables,
                             self._learner_model.variables)

      variable_client.update()

      # Make sure no new requests are made.
      self.assertEqual(variable_client._future, future)

    # Calling wait() on the barrier will now allow the variables to be copied
    # over from source to client.
    barrier.wait()

    # Update once more to ensure the variables are copied over.
    while variable_client._future is not None:
      variable_client.update()

    # After a number of update calls, the variables should be the same.
    self.assertAllClose(self._actor_model.variables,
                        self._learner_model.variables)


if __name__ == '__main__':
  tf.test.main()
